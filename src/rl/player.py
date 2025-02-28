import random

_players = ['player_0', 'player_1']


class Players:
    def __init__(self, p=None):
        if p is None:
            self.me_n, self.opp_n, self.me, self.opp = self._reset()
        else:
            self.me_n = 0 if p == 'player_0' else 1
            self.opp_n = 0 if self.me_n == 1 else 1

            self.me = _players[self.me_n]
            self.opp = _players[self.opp_n]

    @staticmethod
    def _reset():
        me_n = random.randint(0, 1)
        # me_n=0
        opp_n = 0 if me_n == 1 else 1

        me = _players[me_n]
        opp = _players[opp_n]

        return me_n, opp_n, me, opp

    def reset(self):
        self.me_n, self.opp_n, self.me, self.opp = self._reset()
