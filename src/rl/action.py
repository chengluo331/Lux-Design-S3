import numpy as np
from gymnasium.spaces import Box, Discrete, Dict, Tuple, Space, MultiDiscrete

from luxai_s3.params import env_params_ranges


class Action:
    def __init__(self, env_params):
        self._env_params = env_params
        # self._offset = np.tile([0, env_params_ranges["unit_sap_range"][-1],
        #                         env_params_ranges["unit_sap_range"][-1]],
        #                        (self._env_params.max_units, 1))

    def get_space(self):
        max_unit = self._env_params.max_units
        # low = -env_params_ranges["unit_sap_range"][-1]
        # high = env_params_ranges["unit_sap_range"][-1]
        # return MultiDiscrete([6, high - low + 1, high - low + 1] * max_unit, start=[0, 0, 0] * max_unit)
        return MultiDiscrete([5] * max_unit)

    # get lux env action
    def get_action(self, action):
        # return action.reshape(self._env_params.max_units, -1)-self._offset
        act = np.zeros((self._env_params.max_units, 3), dtype=np.int16)
        act[:,0] = action
        return act

    def reset(self):
        pass
