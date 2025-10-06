import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, pooled_shape, altitudes, features_dim=256):
        """
        observation_space: gym.spaces.Box (1D vector space)
        pooled_shape: (H, W) -> size of the patch after pooling
        altitudes: number of altitude levels
        features_dim: final dimension of the extracted features
        """
        super().__init__(observation_space, features_dim)

        self.H, self.W = pooled_shape
        self.altitudes = altitudes
        self.pooled_size = self.H * self.W

        # Define CNN to process the image-like part of the observation
        in_channels = 1 + altitudes
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, in_channels, self.H, self.W)
            n_flat = self.cnn(sample).shape[1]

        # Fully connected layer to combine CNN output with one-hot altitude encoding
        self.linear = nn.Sequential(
            nn.Linear(n_flat + altitudes, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, obs):
        # Splitting observation
        batch_size = obs.shape[0]
        belief = obs[:, :self.pooled_size]  # [B, pooled_size]
        counts = obs[:, self.pooled_size:self.pooled_size*(1+self.altitudes)]
        one_hot = obs[:, -self.altitudes:]  # [B, altitudes]

        belief_img = belief.view(batch_size, 1, self.H, self.W)
        counts_img = counts.view(batch_size, self.altitudes, self.H, self.W)

        x_img = torch.cat([belief_img, counts_img], dim=1)  # [B, 1+A, H, W]

        cnn_out = self.cnn(x_img)

        x = torch.cat([cnn_out, one_hot], dim=1)

        return self.linear(x)
