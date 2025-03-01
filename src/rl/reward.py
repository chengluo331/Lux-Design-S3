import numpy as np
from rl.observation import get_obs
from scipy.ndimage import convolve


class Reward:
    def __init__(self, players, env_params):
        self.players = players
        self.env_params = env_params

    def reset(self, players):
        self.players = players

    def calculate(self, last_obs, current_obs):
        result = 0.
        obs_player = current_obs[self.players.me]
        last_obs_player = last_obs[self.players.me]

        # reward team win, penalise team loss
        pre_team_wins = last_obs_player['team_wins'][self.players.me_n]
        new_team_wins = obs_player['team_wins'][self.players.me_n]
        result += (new_team_wins - pre_team_wins)

        pre_opp_team_wins = last_obs_player['team_wins'][self.players.opp_n]
        new_opp_team_wins = obs_player['team_wins'][self.players.opp_n]
        result -= (new_opp_team_wins - pre_opp_team_wins)

        # # reward team point
        pre_team_points = last_obs_player['team_points'][self.players.me_n]
        new_team_points = obs_player['team_points'][self.players.me_n]
        result += (new_team_points - pre_team_points)*0.1

        # reward exploration
        unit_exists_mask = obs_player['units_mask'][self.players.me_n] & last_obs_player['units_mask'][
            self.players.me_n]
        moved_mask = obs_player['units']['position'][self.players.me_n] != last_obs_player['units']['position'][
            self.players.me_n]
        result += sum(moved_mask.any(axis=1) * unit_exists_mask) * 0.0001

        # reward increasing visibility (about 0.3 per game)
        last_obs_dict = get_obs(self.players, last_obs_player, self.env_params)
        current_obs_dict = get_obs(self.players, obs_player, self.env_params)

        width = self.env_params.map_width
        height = self.env_params.map_height
        current_visibility = np.sum(current_obs_dict['visibility'])
        last_visibility = np.sum(last_obs_dict['visibility'])
        if last_visibility > 0:
            result += (current_visibility - last_visibility) / (width * height)

        # reward discovering relic
        current_relics = np.sum(current_obs_dict['relic_nodes'])
        last_relics = np.sum(last_obs_dict['relic_nodes'])
        if current_relics > last_relics:
            result += (current_relics - last_relics) / self.env_params.max_relic_nodes

        # reward staying around relic
        kernel = np.ones((5, 5), dtype=np.int8)

        last_convolved = convolve(last_obs_dict['relic_nodes'], kernel, mode='constant', cval=0)
        last_points_range = (last_convolved > 0).astype(np.int8)
        last_unit_points = np.sum(last_obs_dict['units_position'] * last_points_range)

        convolved = convolve(current_obs_dict['relic_nodes'], kernel, mode='constant', cval=0)
        points_range = (convolved > 0).astype(np.int8)
        current_unit_points = np.sum(current_obs_dict['units_position'] * points_range)
        if current_unit_points > last_unit_points:
            result += (current_unit_points - last_unit_points)

        # reward energy node
        current_energy = np.sum(current_obs_dict['map_features_energy'] * current_obs_dict['units_position'])
        last_energy = np.sum(last_obs_dict['map_features_energy'] * last_obs_dict['units_position'])
        result += (current_energy - last_energy) * 0.01

        # reward/penalise unit energy
        unit_energy_diff = np.sum(current_obs_dict["units_energy"] - last_obs_dict["units_energy"])
        result += unit_energy_diff * 0.01

        return result
