class Players:
    def __init__(self):
        self.me = 'player_0'
        self.opp = "player_1" if self.me == "player_0" else "player_0"

        self.me_n = 0 if self.me == "player_0" else 1
        self.opp_n = 0 if self.opp == "player_0" else 1

    def reset(self):
        pass
