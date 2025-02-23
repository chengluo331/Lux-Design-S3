import numpy as np
from gymnasium.spaces import Box, Discrete, Dict, Tuple, Space, MultiDiscrete, MultiBinary


class Observation:
    def __init__(self, players, env_params):
        self._players = players
        self._env_params = env_params

    def get_space(self):
        params = self._env_params
        width = params.map_width
        height = params.map_height
        num_teams = params.num_teams
        max_units = params.max_units
        max_relics = params.max_relic_nodes
        min_energy = params.min_unit_energy
        max_energy = params.max_unit_energy
        min_energy_tile = params.min_energy_per_tile
        max_energy_tile = params.max_energy_per_tile
        match_per_episode = params.match_count_per_episode
        max_steps = params.max_steps_in_match

        return Dict({
            "units_position": MultiBinary((num_teams, width, height)),
            "units_energy": Box(low=0., high=1., shape=(num_teams, width, height)),
            # "units_mask": Box(low=0, high=1, shape=(num_teams, max_units), dtype=np.int8),
            # "sensor_mask": Box(low=0, high=1, shape=(width, height), dtype=np.int8),
            "map_features_energy": Box(low=0., high=1., shape=(width, height)),
            "nebula": MultiBinary((width, height)),
            "asteroid": MultiBinary((width, height)),
            # "relic_nodes_mask": Box(low=0, high=1, shape=(max_relics,), dtype=np.int32),
            "relic_nodes": MultiBinary((width, height)),
            "visibility": MultiBinary((width, height)),
            # "team_points": gspc.Box(low=0, high=max_points, shape=(num_teams,), dtype=np.int32),
            # "team_wins": gspc.Box(low=0, high=match_per_episode, shape=(num_teams,), dtype=np.int32),
            # "steps": gspc.Discrete(match_per_episode * (max_steps + 1) + 1),  # Assuming an upper limit for steps
            # "match_steps": gspc.Discrete(max_steps + 1)
        })

    def get_obs(self, obs):
        my_obs = obs[self._players.me]

        params = self._env_params
        width = params.map_width
        height = params.map_height
        num_teams = params.num_teams
        max_units = params.max_units
        max_relics = params.max_relic_nodes
        min_energy = params.min_unit_energy
        max_energy = params.max_unit_energy
        min_energy_tile = params.min_energy_per_tile
        max_energy_tile = params.max_energy_per_tile
        match_per_episode = params.match_count_per_episode
        max_steps = params.max_steps_in_match

        # init obs
        units_position = np.zeros((num_teams, width, height), dtype=np.int8)
        units_energy = np.zeros((num_teams, width, height), dtype=np.float32)
        map_features_energy = np.zeros((width, height), dtype=np.float32)
        # nebula = np.zeros((width, height), dtype=np.int8),
        # asteroid = np.zeros((width, height), dtype=np.int8),
        relic_nodes = np.zeros((width, height), dtype=np.int8)
        # visibility = np.zeros((width, height), dtype = np.int8),

        # set obs
        u_mask = my_obs['units_mask']
        pos = my_obs['units']['position']
        energy = my_obs['units']['energy']

        me_n = self._players.me_n
        opp_n = self._players.opp_n
        # units
        energy_range = max_energy - min_energy
        for u in range(max_units):
            for t in (me_n, opp_n):
                if u_mask[t][u]:
                    units_position[t, pos[t][u]] = 1
                    units_energy[t, pos[t][u]] = (energy[t][u] - min_energy) / energy_range

        visibility = my_obs['sensor_mask']

        tile_energy_range = max_energy_tile - min_energy_tile
        map_features_energy=(map_features_energy-min_energy_tile)/tile_energy_range
        map_features_energy*=visibility

        nebula = (my_obs['map_features']['tile_type']==1)
        asteroid = (my_obs['map_features']['tile_type']==2)

        for r in range(max_relics):
            if my_obs['relic_nodes_mask'][r]:
                relic_nodes[my_obs['relic_nodes'][r]] = 1

        return {
            "units_position": units_position,
            "units_energy": units_energy,
            "map_features_energy": map_features_energy,
            "nebula": nebula,
            "asteroid": asteroid,
            "relic_nodes": relic_nodes,
            "visibility": visibility
        }

    def reset(self):
        pass
