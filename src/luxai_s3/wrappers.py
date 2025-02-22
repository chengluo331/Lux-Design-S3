# TODO (stao): Add lux ai s3 env to gymnax api wrapper, which is the old gym api
import json
import os
from typing import Any, SupportsFloat
import flax
import flax.serialization
import gymnasium as gym
import gymnax
import gymnax.environments.spaces
from gymnasium.spaces import MultiDiscrete
import jax
import numpy as np
import dataclasses
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.state import serialize_env_actions, serialize_env_states
from luxai_s3.utils import to_numpy
import jax.numpy as jnp
from gymnax.environments.spaces import Box, Discrete, Dict, Tuple, Space

gspc = gym.spaces


def gymnax_space_to_gym_space(space: Space) -> gspc.Space:
    """Convert Gymnax space to equivalent Gym space."""
    if isinstance(space, Discrete):
        return gspc.Discrete(space.n)
    elif isinstance(space, Box):
        low = (
            float(space.low)
            if (np.isscalar(space.low) or space.low.size == 1)
            else np.array(space.low)
        )
        high = (
            float(space.high)
            if (np.isscalar(space.high) or space.low.size == 1)
            else np.array(space.high)
        )
        return gspc.Box(low, high, space.shape, space.dtype)
    elif isinstance(space, Dict):
        return gspc.Dict({k: gymnax_space_to_gym_space(v) for k, v in space.spaces.items()})
    elif isinstance(space, Tuple):
        return gspc.Tuple(space.spaces)
    else:
        raise NotImplementedError(
            f"Conversion of {space.__class__.__name__} not supported"
        )


class LuxAIS3GymEnv(gym.Env):
    def __init__(self, numpy_output: bool = False):
        self.numpy_output = numpy_output
        self.rng_key = jax.random.key(0)
        self.jax_env = LuxAIS3Env(auto_reset=False)
        self.env_params: EnvParams = EnvParams()

        low = np.zeros((self.env_params.max_units, 3))
        low[:, 1:] = -self.env_params.unit_sap_range
        high = np.ones((self.env_params.max_units, 3)) * 5
        high[:, 1:] = self.env_params.unit_sap_range
        self.action_space = gym.spaces.Dict(
            dict(
                player_0=gym.spaces.Box(low=low, high=high, dtype=np.int16),
                player_1=gym.spaces.Box(low=low, high=high, dtype=np.int16),
            )
        )

    def render(self):
        self.jax_env.render(self.state, self.env_params)

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            self.rng_key = jax.random.key(seed)
        self.rng_key, reset_key = jax.random.split(self.rng_key)
        # generate random game parameters
        # TODO (stao): check why this keeps recompiling when marking structs as static args
        randomized_game_params = dict()
        for k, v in env_params_ranges.items():
            self.rng_key, subkey = jax.random.split(self.rng_key)
            randomized_game_params[k] = jax.random.choice(
                subkey, jax.numpy.array(v)
            ).item()
        params = EnvParams(**randomized_game_params)
        if options is not None and "params" in options:
            params = options["params"]

        self.env_params = params
        obs, self.state = self.jax_env.reset(reset_key, params=params)
        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))

        # only keep the following game parameters available to the agent
        params_dict = dataclasses.asdict(params)
        params_dict_kept = dict()
        for k in [
            "max_units",
            "match_count_per_episode",
            "max_steps_in_match",
            "map_height",
            "map_width",
            "num_teams",
            "unit_move_cost",
            "unit_sap_cost",
            "unit_sap_range",
            "unit_sensor_range",
        ]:
            params_dict_kept[k] = params_dict[k]
        return obs, dict(
            params=params_dict_kept, full_params=params_dict, state=self.state
        )

    def step(
            self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.rng_key, step_key = jax.random.split(self.rng_key)
        obs, self.state, reward, terminated, truncated, info = self.jax_env.step(
            step_key, self.state, action, self.env_params
        )
        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))
            reward = to_numpy(reward)
            terminated = to_numpy(terminated)
            truncated = to_numpy(truncated)
            # info = to_numpy(flax.serialization.to_state_dict(info))
        return obs, reward, terminated, truncated, info


# TODO: vectorized gym wrapper

class RewardSpace:
    def __init__(self, player):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.player_n = 0 if self.player == "player_0" else 1
        self.team_wins = 0.
        self.team_points = 0.

    def reset(self):
        self.team_wins = 0.
        self.team_points = 0.

    def calculate(self, obs):
        result = 0.
        obs_player = obs[self.player]

        new_team_points = obs_player['team_points'][self.player_n]
        result += (new_team_points - self.team_points)
        self.team_points = new_team_points

        new_team_wins = obs_player['team_wins'][self.player_n]
        result += (new_team_wins - self.team_wins) * 100.
        self.team_wins = new_team_wins

        return result


class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env: LuxAIS3GymEnv, player: str, opp_agent_class):
        super().__init__(env)
        # original action space from lux env but box space gives floating number in RL training
        # self.action_space = self.env.action_space[player]
        unwrapped_env = self.env.unwrapped
        unit_sap_range = unwrapped_env.env_params.unit_sap_range
        self.action_space = MultiDiscrete([5, unit_sap_range, unit_sap_range] * unwrapped_env.env_params.max_units,
                                          dtype=np.int16)

        self.observation_space = self._get_observation_space()
        self._metadata = unwrapped_env.metadata

        self.player = player
        self._reward_space = RewardSpace(player)

        # TODO: replace opponent agent with self trained
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.opp_agent_class = opp_agent_class
        self.opp_agent = None
        self.steps = 0
        self.last_obs = None

    def _get_observation_space(self):
        params = self.env.unwrapped.env_params
        width = params.map_width
        height = params.map_height
        num_teams = params.num_teams
        max_units = params.max_units
        max_relics = params.max_relic_nodes
        # min_energy = params.min_unit_energy
        max_energy = params.max_unit_energy
        min_energy_tile = params.min_energy_per_tile
        max_energy_tile = params.max_energy_per_tile
        max_points = 1000
        match_per_episode = params.match_count_per_episode
        max_steps = params.max_steps_in_match

        # gspc.Dict({
        #     "units_position": gspc.Box(low=-1, high=max(width, height) - 1, shape=(num_teams, max_units, 2),
        #                                dtype=np.int32),
        #     "units_energy": gspc.Box(low=-1, high=max_energy, shape=(num_teams, max_units), dtype=np.int32),
        #     "units_mask": gspc.Box(low=0, high=1, shape=(num_teams, max_units), dtype=np.int8),
        #     "sensor_mask": gspc.Box(low=0, high=1, shape=(width, height), dtype=np.int8),
        #     "map_features_energy": gspc.Box(low=min(-1, min_energy_tile), high=max_energy_tile,
        #                                     shape=(width, height),
        #                                     dtype=np.float32),
        #     "map_features_tile_type": gspc.Box(low=-1, high=2, shape=(width, height), dtype=np.int32),
        #     "relic_nodes_mask": gspc.Box(low=0, high=1, shape=(max_relics,), dtype=np.int32),
        #     "relic_nodes": gspc.Box(low=-1, high=max(width, height) - 1, shape=(max_relics, 2), dtype=np.int32),
        #     "team_points": gspc.Box(low=0, high=max_points, shape=(num_teams,), dtype=np.int32),
        #     "team_wins": gspc.Box(low=0, high=match_per_episode, shape=(num_teams,), dtype=np.int32),
        #     "steps": gspc.Discrete(match_per_episode * (max_steps + 1) + 1),  # Assuming an upper limit for steps
        #     "match_steps": gspc.Discrete(max_steps + 1)
        # })
        return gspc.Dict({
            "units_position": gspc.Box(low=-1, high=max(width, height) - 1, shape=(num_teams, max_units, 2),
                                       dtype=np.int32),
            "units_energy": gspc.Box(low=-1, high=max_energy, shape=(num_teams, max_units), dtype=np.int32),
            "units_mask": gspc.Box(low=0, high=1, shape=(num_teams, max_units), dtype=np.int8),
            "sensor_mask": gspc.Box(low=0, high=1, shape=(width, height), dtype=np.int8),
            "map_features_energy": gspc.Box(low=min(-1, min_energy_tile), high=max_energy_tile,
                                            shape=(width, height),
                                            dtype=np.float32),
            "map_features_tile_type": gspc.Box(low=-1, high=2, shape=(width, height), dtype=np.int32),
            "relic_nodes_mask": gspc.Box(low=0, high=1, shape=(max_relics,), dtype=np.int32),
            "relic_nodes": gspc.Box(low=-1, high=max(width, height) - 1, shape=(max_relics, 2), dtype=np.int32),
            "team_points": gspc.Box(low=0, high=max_points, shape=(num_teams,), dtype=np.int32),
            "team_wins": gspc.Box(low=0, high=match_per_episode, shape=(num_teams,), dtype=np.int32),
            "steps": gspc.Discrete(match_per_episode * (max_steps + 1) + 1),  # Assuming an upper limit for steps
            "match_steps": gspc.Discrete(max_steps + 1)
        })

    # Taking OBS from lux env (original obs) and transform to obs for training
    def _env_obs_to_my_obs(self, obs):
        obs_player = obs[self.player]
        return {
            'units_position': obs_player['units']['position'],
            'units_energy': obs_player['units']['energy'],
            'units_mask': obs_player['units_mask'],
            'sensor_mask': obs_player['sensor_mask'],
            'map_features_energy': obs_player['map_features']['energy'],
            'map_features_tile_type': obs_player['map_features']['tile_type'],
            'relic_nodes_mask': obs_player['relic_nodes_mask'],
            'relic_nodes': obs_player['relic_nodes'],
            'team_points': obs_player['team_points'],
            'team_wins': obs_player['team_wins'],
            'steps': obs_player['steps'].tolist(),
            'match_steps': obs_player['match_steps'].tolist(),
        }

    # Taking obs for training and transform to lux env original obs (for testing purpose)
    def backout_obs(self, obs):
        team_id = 0 if self.player == "player_0" else 1

        def _(v):
            return [v, 0] if team_id == 0 else [0, v]

        return {
            'units': {
                'position': obs['units_position'],
                'energy': obs['units_energy']
            },
            'units_mask': obs['units_mask'],
            'sensor_mask': obs['sensor_mask'],
            'map_features': {
                'energy': obs['map_features_energy'],
                'tile_type': obs['map_features_tile_type']
            },
            'relic_nodes_mask': obs['relic_nodes_mask'],
            'relic_nodes': obs['relic_nodes'],
            'team_points': obs['team_points'],
            'team_wins': obs['team_wins'],
            'steps': obs['steps'],
            'match_steps': obs['match_steps'],
        }

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._reward_space.reset()

        self.opp_agent = self.opp_agent_class(player=self.opp_player, env_cfg=info["params"])
        self.steps = 0
        self.last_obs = obs[self.opp_player]
        return self._env_obs_to_my_obs(obs), info

    def step(
            self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        # my_action = action
        # for RL training, action_space is multidiscrete so we need to transform back to the format lux env
        # can understand
        my_action = action.reshape(self.env.unwrapped.env_params.max_units, -1)

        opp_action = self.opp_agent.act(self.steps, self.last_obs)
        obs, _, terminated, truncated, info = self.env.step(
            {self.player: my_action, self.opp_player: opp_action}
        )
        self.steps += 1
        self.last_obs = obs[self.opp_player]

        return (self._env_obs_to_my_obs(obs),
                self._reward_space.calculate(obs),
                terminated[self.player].tolist(),
                truncated[self.player].tolist(),
                info)


class RecordEpisode(gym.Wrapper):
    def __init__(
            self,
            env: LuxAIS3GymEnv,
            save_dir: str = None,
            save_on_close: bool = True,
            save_on_reset: bool = True,
    ):
        super().__init__(env)
        self.episode = dict(states=[], actions=[], metadata=dict())
        self.episode_id = 0
        self.save_dir = save_dir
        self.save_on_close = save_on_close
        self.save_on_reset = save_on_reset
        self.episode_steps = 0
        if save_dir is not None:
            from pathlib import Path

            Path(save_dir).mkdir(parents=True, exist_ok=True)

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if self.save_on_reset and self.episode_steps > 0:
            self._save_episode_and_reset()
        obs, info = self.env.reset(seed=seed, options=options)

        self.episode["metadata"]["seed"] = seed
        self.episode["params"] = flax.serialization.to_state_dict(info["full_params"])
        self.episode["states"].append(info["state"])
        return obs, info

    def step(
            self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.episode["states"].append(info["final_state"])
        self.episode["actions"].append(action)
        return obs, reward, terminated, truncated, info

    def serialize_episode_data(self, episode=None):
        if episode is None:
            episode = self.episode
        ret = dict()
        ret["observations"] = serialize_env_states(episode["states"])
        if "actions" in episode:
            ret["actions"] = serialize_env_actions(episode["actions"])
        ret["metadata"] = episode["metadata"]
        ret["params"] = episode["params"]
        return ret

    def save_episode(self, save_path: str):
        episode = self.serialize_episode_data()
        with open(save_path, "w") as f:
            json.dump(episode, f)
        self.episode = dict(states=[], actions=[], metadata=dict())

    def _save_episode_and_reset(self):
        """saves to generated path based on self.save_dir and episoe id and updates relevant counters"""
        self.save_episode(
            os.path.join(self.save_dir, f"episode_{self.episode_id}.json")
        )
        self.episode_id += 1
        self.episode_steps = 0

    def close(self):
        if self.save_on_close and self.episode_steps > 0:
            self._save_episode_and_reset()
