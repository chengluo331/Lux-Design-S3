import numpy as np
from rl.observation import get_obs
from scipy.ndimage import convolve

w_win = 0.125
w_points = 0.125
w_vis = 0.125
w_relics = 0.125

w_u_explore = 0.125
w_u_relic = 0.125
w_u_nebula = 0.125
w_u_energy = 0.125


class stats:
    def __init__(self):
        self.n = 0
        self.u = 0.
        self.s2 = 0.

    def update(self, x):
        self.n += 1
        d1 = x - self.u
        self.u += d1 / self.n
        d2 = x - self.u
        self.s2 += d1 * d2

        return (x - self.u) / (self.std() + 1e-8)

    def std(self):
        return np.sqrt(self.s2 / self.n) if self.n > 1 else 1.


class Reward:
    def __init__(self, players, env_params):
        self.players = players
        self.env_params = env_params

        self.win_stats = stats()
        self.points_stats = stats()
        self.vis_stats = stats()
        self.relics_stats = stats()

        self.u_explore_stats = stats()
        self.u_relic_stats = stats()
        self.u_nebula_stats = stats()
        self.u_energy_stats = stats()

    def reset(self, players):
        self.players = players

        self.win_stats = stats()
        self.points_stats = stats()
        self.vis_stats = stats()
        self.relics_stats = stats()

        self.u_explore_stats = stats()
        self.u_relic_stats = stats()
        self.u_nebula_stats = stats()
        self.u_energy_stats = stats()

    # TODO
    # exploration + vis
    # energy
    # relic
    # points
    # win
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
        match_steps = current_obs[ts]['match_steps'].tolist()

        if match_steps>0:
            # # reward team point
            pre_team_points = last_obs_player['team_points'][t]
            new_team_points = obs_player['team_points'][t]
            rew_team_points = (new_team_points - pre_team_points)
            if rew_team_points > 0:
                rew_team_points = self.points_stats.update(rew_team_points) * w_points
                result += rew_team_points

            # reward increasing visibility (about 0.3 per game)
            last_obs_dict = get_obs(self.players, last_obs_player, self.env_params)
            current_obs_dict = get_obs(self.players, obs_player, self.env_params)

            width = self.env_params.map_width
            height = self.env_params.map_height
            current_visibility = np.sum(current_obs_dict['visibility'])
            last_visibility = np.sum(last_obs_dict['visibility'])
            rew_vis = (current_visibility - last_visibility) / (width * height) if last_visibility > 0 else 0
            rew_vis = self.vis_stats.update(rew_vis) * w_vis
            result += rew_vis

            # reward discovering relic
            current_relics = np.sum(current_obs_dict['relic_nodes'])
            last_relics = np.sum(last_obs_dict['relic_nodes'])
            rew_relics_disc = (current_relics - last_relics) / self.env_params.max_relic_nodes
            if rew_relics_disc != 0:
                rew_relics_disc = self.relics_stats.update(rew_relics_disc) * w_relics
                result += rew_relics_disc

            ########################################################################
            # for each unit:
            #     if exists in the last round
            #         explore reward
            #         approaching relic reward
            #         approaching energy tile reward ?
            #         collect energy
            #         penalise on nebula

            unit_exists_mask = obs_player['units_mask'][t] & last_obs_player['units_mask'][t]

            # reward exploration
            moved_mask = obs_player['units']['position'][t] != last_obs_player['units']['position'][t]
            rew_explore = sum(np.any(moved_mask, axis=1) * unit_exists_mask)
            rew_explore = self.u_explore_stats.update(rew_explore) * w_u_explore
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
                        rew_nebula -= 1. / max_units

            if np.any(relic_mask):
                rew_approaching_relic = self.u_relic_stats.update(rew_approaching_relic) * w_u_relic
                result += rew_approaching_relic

            if rew_nebula > 0.:
                rew_nebula = self.u_nebula_stats.update(rew_nebula) * w_u_nebula
                result += rew_nebula

            # collect energy
            energy_diff = (obs_player['units']['energy'][t] - last_obs_player['units']['energy'][t]) * unit_exists_mask
            rew_collect_energy = np.sum(
                unit_exists_mask * energy_diff * (energy_diff > 0)) / self.env_params.max_unit_energy

            if rew_collect_energy > 0.0:
                rew_collect_energy = self.u_energy_stats.update(rew_collect_energy) * w_u_energy
                result += rew_collect_energy

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

            # print(f"""
            # rew_team_points: {rew_team_points}
            # rew_vis: {rew_vis}
            # rew_relics_disc: {rew_relics_disc}
            #
            # rew_approaching_relic: {rew_approaching_relic}
            # rew_explore: {rew_explore}
            # rew_collect_energy: {rew_collect_energy}
            # rew_nebula: {rew_nebula}
            # """)
        else:
            # reward team win, penalise team loss
            pre_team_wins = last_obs_player['team_wins'][t]
            new_team_wins = obs_player['team_wins'][t]

            pre_opp_team_wins = last_obs_player['team_wins'][self.players.opp_n]
            new_opp_team_wins = obs_player['team_wins'][self.players.opp_n]
            rew_win = (new_team_wins - pre_team_wins) - (new_opp_team_wins - pre_opp_team_wins)
            if rew_win != 0:
                rew_win = self.win_stats.update(rew_win) * w_win
                result += rew_win

        return result
