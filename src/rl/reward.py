import numpy as np
from rl.observation import get_obs
from scipy.ndimage import convolve


def distance_to_relics(obs_player, me_n):
    d = 0.
    relics = [loc for loc in obs_player['relic_nodes'] if loc[0] != -1]
    if relics:
        a, b = np.average(relics, axis=0)
        n = 0
        for x, y in obs_player['units']['position'][me_n]:
            if x > 0 and y > 0:
                n += 1
                d += (abs(x - a) + abs(y - b)) / 24.
        if n > 0:
            d /= n
    return d



class Reward:
    def __init__(self, players, env_params):
        self.players = players
        self.env_params = env_params

    def reset(self, players):
        self.players = players

    def calculate(self, last_obs, current_obs):
        max_units = self.env_params.max_units
        t = self.players.me_n
        ts = self.players.me
        # for team:
        #     reward/penalise team win/loss
        #     reward team point
        #     reward/penalise team visibility
        #     collecting energy reward
        result = 0.
        obs_player = current_obs[ts]
        last_obs_player = last_obs[ts]

        # reward team win, penalise team loss
        pre_team_wins = last_obs_player['team_wins'][t]
        new_team_wins = obs_player['team_wins'][t]
        result += (new_team_wins - pre_team_wins)

        pre_opp_team_wins = last_obs_player['team_wins'][self.players.opp_n]
        new_opp_team_wins = obs_player['team_wins'][self.players.opp_n]
        result -= (new_opp_team_wins - pre_opp_team_wins)

        # # reward team point
        pre_team_points = last_obs_player['team_points'][t]
        new_team_points = obs_player['team_points'][t]
        rew_team_points = 0.1 * (new_team_points - pre_team_points)
        result += rew_team_points

        # reward increasing visibility (about 0.3 per game)
        last_obs_dict = get_obs(self.players, last_obs_player, self.env_params)
        current_obs_dict = get_obs(self.players, obs_player, self.env_params)

        width = self.env_params.map_width
        height = self.env_params.map_height
        current_visibility = np.sum(current_obs_dict['visibility'])
        last_visibility = np.sum(last_obs_dict['visibility'])
        rew_vis = 10. * (current_visibility - last_visibility) / (width * height) if last_visibility > 0 else 0
        result += rew_vis

        # reward discovering relic
        current_relics = np.sum(current_obs_dict['relic_nodes'])
        last_relics = np.sum(last_obs_dict['relic_nodes'])
        rew_relics_disc = (current_relics - last_relics) / self.env_params.max_relic_nodes
        result += rew_relics_disc

        ########################################################################
        # for each unit:
        #     if exists in the last round
        #         explore reward
        #         approaching relic reward
        #         approaching energy tile reward ?
        #         collect energy
        #         penalise on nebula
        #         penalise removal

        unit_exists_mask = obs_player['units_mask'][t] & last_obs_player['units_mask'][t]

        # reward exploration
        moved_mask = obs_player['units']['position'][t] != last_obs_player['units']['position'][t]
        rew_explore = sum(np.any(moved_mask, axis=1) * unit_exists_mask) * 0.001
        result += rew_explore

        # relic
        max_relics = self.env_params.max_relic_nodes
        relic_mask = np.any(obs_player['relic_nodes'] != -1, axis=1) & np.any(last_obs_player['relic_nodes'] != -1,
                                                                              axis=1)

        rew_approaching_relic = 0.
        rew_nebula = 0.
        for u in range(max_units):
            # exist in both the last and current step
            if unit_exists_mask[u]:
                u_x, u_y = obs_player['units']['position'][t][u]
                u_x0, u_y0 = last_obs_player['units']['position'][t][u]

                # approaching relic reward
                for r in range(max_relics):
                    if relic_mask[r]:
                        r_x, r_y = obs_player['relic_nodes'][r]
                        rew_approaching_relic += ((abs(u_x0 - r_x) + abs(u_y0 - r_y)) - (
                                abs(u_x - r_x) + abs(u_y - r_y))) / width

                # penalise nebula
                if current_obs_dict['nebula'][u_x, u_y]:
                    rew_nebula += 0.1 * 1. / max_units

        result += rew_approaching_relic
        result -= rew_nebula

        # collect energy
        energy_diff = (obs_player['units']['energy'][t] - last_obs_player['units']['energy'][t]) * unit_exists_mask
        rew_collect_energy = np.sum(energy_diff * (energy_diff > 0)) / self.env_params.max_unit_energy
        result += rew_collect_energy

        # penalise removal
        rew_removal = 0.1 * np.sum(~obs_player['units_mask'][t] & last_obs_player['units_mask'][t]) / max_units
        result -= rew_removal

        # reward staying around relic
        # kernel = np.ones((5, 5), dtype=np.int8)
        #
        # last_convolved = convolve(last_obs_dict['relic_nodes'], kernel, mode='constant', cval=0)
        # last_points_range = (last_convolved > 0).astype(np.int8)
        # last_unit_points = np.sum(last_obs_dict['units_position'][self.players.me_n] * last_points_range)
        #
        # convolved = convolve(current_obs_dict['relic_nodes'], kernel, mode='constant', cval=0)
        # points_range = (convolved > 0).astype(np.int8)
        # current_unit_points = np.sum(current_obs_dict['units_position'][self.players.me_n] * points_range)
        # if current_unit_points > last_unit_points:
        #     result += (current_unit_points - last_unit_points)

        return result
