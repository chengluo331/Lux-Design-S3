import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

class CustomBoardFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0

        for key, space in observation_space.spaces.items():
            # Determine the number of channels:
            # For Box: assume shape is (n_teams, width, height)
            # For MultiBinary: assume shape is (width, height) -> 1 channel
            if key.startswith('units_'):
                channels = space.shape[0]
                height, width = space.shape[1], space.shape[2]
            else:
                channels = 1
                height, width = space.shape[0], space.shape[1]

            # Create a simple CNN for this key:
            # cnn_extractor = nn.Sequential(
            #     nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(),
            #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(),
            #     nn.Flatten()
            # )
            cnn_extractor = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample
                nn.Flatten()
            )

            extractors[key] = cnn_extractor
            # Compute the output dimension for this key using a dummy tensor
            with th.no_grad():
                dummy_input = th.zeros(1, channels, height, width)
                output_dim = cnn_extractor(dummy_input).shape[1]
            total_concat_size += output_dim

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
        # Optional: a final fully connected layer to get the desired features_dim
        # self.fc = nn.Sequential(
        #     nn.Linear(total_concat_size, features_dim),
        #     nn.ReLU()
        # )

    def forward(self, observations):
        encoded_tensors = []
        for key, extractor in self.extractors.items():
            x = observations[key]
            if x.ndim == 3:  # if shape is (batch, height, width), add channel dim
                x = x.unsqueeze(1)
            x = x.float()
            encoded_tensors.append(extractor(x))
        concatenated = th.cat(encoded_tensors, dim=1)
        return concatenated