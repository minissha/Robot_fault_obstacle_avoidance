"""
Lightweight PointNet-style encoder for Stage 2 (noisy 3D perception).

Design goals: CPU-fast, permutation-invariant, embodiment-agnostic (it only
ever sees points in the world/table frame — it has no idea which gripper is
attached). That last property is what makes it reusable as the *shared*
encoder in Stage 4/5 (cross-embodiment experiment): the same weights can
condition on point clouds captured from either the 2-finger or 3-finger
robot, since geometry, not embodiment identity, drives the representation.

Architecture: per-point MLP (shared weights) -> per-channel max-pool
("symmetric function" from the PointNet paper, simplified: no T-Net,
no local-vs-global feature concat) -> MLP head -> fixed-size embedding.

Input: (B, N, C) point cloud, C = 3 (xyz) or 4 (xyz + binary segmentation
mask marking "this point belongs to the target cube" vs "background/table").
We use C=4 by default since the scripted/PPO policies need a way to know
*which* point cluster is the object worth grasping — that mirrors what a
real object-segmentation front end would hand you.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PointNetLite(nn.Module):
    def __init__(self, in_channels: int = 4, embed_dim: int = 128,
                 point_hidden: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, point_hidden),
            nn.ReLU(),
            nn.Linear(point_hidden, point_hidden),
            nn.ReLU(),
            nn.Linear(point_hidden, embed_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, points: torch.Tensor, mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """
        points: (B, N, C) padded point cloud (dropout/occlusion may leave
                fewer real points; padded slots should be all-zero AND
                flagged in `mask`).
        mask:   (B, N) 1.0 = real point, 0.0 = padding. If None, all points
                are treated as real (use only when N is fixed / no dropout).
        returns: (B, embed_dim) global feature.
        """
        feats = self.point_mlp(points)  # (B, N, embed_dim)
        if mask is not None:
            neg_inf = torch.finfo(feats.dtype).min
            feats = feats.masked_fill(mask.unsqueeze(-1) == 0, neg_inf)
        pooled, _ = feats.max(dim=1)  # symmetric fn -> permutation invariant
        pooled = torch.nan_to_num(pooled, neginf=0.0)  # all-padding safety
        return self.head(pooled)


class PointCloudStateEncoder(nn.Module):
    """
    Full Stage-2 observation encoder: PointNet-lite over the scene point
    cloud, concatenated with the small amount of proprioception every robot
    embodiment can cheaply self-report (ee pose + gripper state) that we do
    NOT ask perception to infer. Outputs a flat feature vector usable as a
    drop-in `features_extractor` for an SB3 policy or as the trunk for BC.
    """

    def __init__(self, n_points: int = 768, point_channels: int = 4,
                 proprio_dim: int = 4, embed_dim: int = 128):
        super().__init__()
        self.n_points = n_points
        self.point_channels = point_channels
        self.proprio_dim = proprio_dim
        self.pointnet = PointNetLite(in_channels=point_channels,
                                      embed_dim=embed_dim)
        self.out_dim = embed_dim + proprio_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs is a flattened vector produced by
        ReachPickPushPerceptionEnv._get_obs():
            [ points.flatten() (n_points * point_channels),
              point_mask (n_points),
              proprio (proprio_dim) ]
        We unflatten it here so the whole thing stays a single Box space
        (simplest to make SB3-compatible without a Dict obs space).
        """
        b = obs.shape[0]
        pts_len = self.n_points * self.point_channels
        pts = obs[:, :pts_len].view(b, self.n_points, self.point_channels)
        mask = obs[:, pts_len:pts_len + self.n_points]
        proprio = obs[:, pts_len + self.n_points:]
        global_feat = self.pointnet(pts, mask)
        return torch.cat([global_feat, proprio], dim=-1)
