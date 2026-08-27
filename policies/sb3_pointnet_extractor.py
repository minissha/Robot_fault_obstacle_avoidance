"""
Adapter so PointCloudStateEncoder can be plugged into SB3's PPO as a
custom `features_extractor_class`, e.g.:

    policy_kwargs = dict(
        features_extractor_class=PointCloudFeaturesExtractor,
        features_extractor_kwargs=dict(n_points=768, point_channels=4,
                                        proprio_dim=4, embed_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)
"""
from __future__ import annotations

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from policies.pointnet_lite import PointCloudStateEncoder


class PointCloudFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, n_points: int = 768,
                 point_channels: int = 4, proprio_dim: int = 4,
                 embed_dim: int = 128):
        encoder = PointCloudStateEncoder(
            n_points=n_points, point_channels=point_channels,
            proprio_dim=proprio_dim, embed_dim=embed_dim,
        )
        super().__init__(observation_space, features_dim=encoder.out_dim)
        self.encoder = encoder

    def forward(self, observations):
        return self.encoder(observations)
