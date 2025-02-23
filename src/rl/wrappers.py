from typing import Any, SupportsFloat

from luxai_s3.wrappers import LuxAIS3GymEnv
import gymnasium as gym

from rl.reward import Reward
from rl.action import Action
from rl.observation import Observation
from rl.player import Players
from kits.python.agent import Agent


class AgentWrapper:
    def __init__(self, agent, obs):
        self._agent = agent
        self.obs = obs
        self.steps = 0

    def get_action(self):
        action = self._agent.act(self.steps, self.obs)
        self.steps += 1
        return action


class RLWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        env_params = self.env.unwrapped.env_params

        self._players = Players()
        self._reward = Reward(self._players, env_params)
        self._action = Action(env_params)
        self._observation = Observation(self._players, env_params)

        self.action_space = self._action.get_space()
        self.observation_space = self._observation.get_space()

        self._opponent_agent = None

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        self._players.reset()
        self._reward.reset()
        self._action.reset()
        self._observation.reset()

        self._opponent_agent = AgentWrapper(
            Agent(self._players.opp, env_cfg=info["params"]),
            obs[self._players.opp]
        )

        self._reward.obs = obs[self._players.me]

        return self._observation.get_obs(obs), info

    def step(
            self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        my_action = self._action.get_action(action)
        # opp_action = self._opponent_agent.get_action()
        opp_action = self._action.get_action(self.action_space.sample())

        me = self._players.me
        opp = self._players.opp

        obs, _, terminated, truncated, info = self.env.step(
            {me: my_action, opp: opp_action}
        )
        reward = self._reward.calculate(obs)

        self._opponent_agent.obs = obs[self._players.opp]
        self._reward.obs = obs[self._players.me]

        return (self._observation.get_obs(obs),
                reward,
                terminated[me].tolist(),
                truncated[opp].tolist(),
                info)
