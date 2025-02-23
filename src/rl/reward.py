class Reward:
    def __init__(self, players, env_params):
        self.players = players
        self.env_params = env_params
        self.obs = None

    def reset(self):
        self.obs = None

    def calculate(self, current_obs):
        result = 0.
        obs_player = current_obs[self.players.me]

        # reward team win, penalise team loss
        pre_team_wins = self.obs['team_wins'][self.players.me_n]
        new_team_wins = obs_player['team_wins'][self.players.me_n]
        result += (new_team_wins - pre_team_wins)

        pre_opp_team_wins = self.obs['team_wins'][self.players.opp_n]
        new_opp_team_wins = obs_player['team_wins'][self.players.opp_n]
        result -= (new_opp_team_wins - pre_opp_team_wins)

        # reward team point
        pre_team_points = self.obs['team_points'][self.players.me_n]
        new_team_points = obs_player['team_points'][self.players.me_n]
        result += (new_team_points - pre_team_points)*0.01

        # reward exploration
        unit_exists_mask = obs_player['units_mask'][self.players.me_n] & self.obs['units_mask'][self.players.me_n]
        moved_mask = obs_player['units']['position'][self.players.me_n] != self.obs['units']['position'][
            self.players.me_n]
        result += sum(moved_mask.any(axis=1) * unit_exists_mask) * 0.001

        # reward collected team energy
        # energy_diff = obs_player['units']['energy'][self.players.me_n] - self.obs['units']['energy'][self.players.me_n]
        # energy_diff_masked = (energy_diff-self.env_params.min_unit_energy)/(self.env_params.max_unit_energy -
        #                                                                     self.env_params.min_unit_energy)*unit_exists_mask
        # result+=sum(energy_diff_masked[energy_diff_masked>0])

        return result