import numpy as np
from gymnasium.spaces import Box, Discrete, Dict, Tuple, Space, MultiDiscrete

class Action:
    def __init__(self, env_params):
        self._env_params = env_params

    def get_space(self):
        return MultiDiscrete([5]* self._env_params.max_units, dtype=np.int16)

    # get lux env action
    def get_action(self, action):
        my_action = np.zeros((self._env_params.max_units, 3), dtype=np.int16)
        my_action[:, 0] = action
        return my_action

    def reset(self):
        pass