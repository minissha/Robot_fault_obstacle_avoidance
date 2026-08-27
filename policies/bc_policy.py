"""Simple MLP behavior-cloning policy for the ground-truth-state Stage 1 task."""
from __future__ import annotations

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int = 19, act_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),  # actions are in [-1, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs) -> "torch.Tensor":
        self.eval()
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            return self.forward(obs).squeeze(0).numpy()
        return self.forward(obs).numpy()


class PointCloudBCPolicy(nn.Module):
    """
    Stage 2/3/4 BC policy: PointCloudStateEncoder trunk (shared-weight
    per-point MLP + max-pool over the noisy point cloud, concatenated with
    proprio/embodiment-onehot) + a small MLP action head. Drop-in
    noisy-observation replacement for `BCPolicy` above.
    """

    def __init__(self, n_points: int, point_channels: int, proprio_dim: int,
                 act_dim: int = 4, embed_dim: int = 128, hidden: int = 128):
        super().__init__()
        from policies.pointnet_lite import PointCloudStateEncoder
        self.encoder = PointCloudStateEncoder(
            n_points=n_points, point_channels=point_channels,
            proprio_dim=proprio_dim, embed_dim=embed_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(obs))

    @torch.no_grad()
    def act(self, obs) -> "torch.Tensor":
        self.eval()
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            return self.forward(obs).squeeze(0).numpy()
        return self.forward(obs).numpy()
