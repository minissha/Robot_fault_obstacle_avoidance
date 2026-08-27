"""
Stage 2 environment: wraps ReachPickPushEnv (ground-truth physics) but
replaces the observation with a SYNTHETIC noisy point cloud instead of
privileged state, to model a real depth-sensor front end.

We don't render actual depth images (slow, and PyBullet's software
rasterizer would dominate CPU time for little research value at this
stage). Instead we synthesize a point cloud directly from ground-truth
mesh/pose knowledge already available in PyBullet:
  - sample points on the cube's surface (object) and on the visible table
    plane (background) in the camera's approximate view frustum
  - apply Gaussian jitter (depth-sensor noise)
  - randomly drop a fraction of points (dropout, e.g. low reflectance)
  - randomly zero out a contiguous angular wedge of points (occlusion, e.g.
    the gripper or arm blocking part of the view)
This keeps everything CPU-only and fast while giving PointNetLite a
realistically corrupted, non-trivial input distribution to be robust to.

Observation layout (flat vector, Box space):
  [ points (n_points * 4: xyz + is_object_mask) flattened,
    point_validity_mask (n_points: 1=real point, 0=padding/dropped),
    proprio (4: ee_xyz, gripper_open) ]

Ground-truth low-dim state is still used internally for reward computation
(same as Stage 1) — only the *observation* the policy sees is degraded.
This mirrors sim-to-real practice: privileged reward at train time, noisy
sensing at both train and deploy time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pybullet as p

from envs.reach_pick_push_env import ReachPickPushEnv, EnvConfig
from envs.embodiments import EmbodimentConfig, GRIPPER_2F, N_EMBODIMENTS


@dataclass
class PerceptionConfig:
    n_points: int = 768
    object_point_frac: float = 0.35     # fraction of points sampled on cube
    jitter_std: float = 0.004           # meters, Gaussian depth noise
    dropout_frac: float = 0.10          # fraction of points randomly killed
    occlusion_prob: float = 0.3         # chance an occlusion wedge occurs
    occlusion_frac: float = 0.20        # fraction of points it can remove
    cube_half_extent: float = 0.025     # cube_small.urdf is ~0.05m per side
    table_sample_radius: float = 0.30   # radius around cube center on table
    seed: int | None = None


class ReachPickPushPerceptionEnv(ReachPickPushEnv):
    def __init__(self, config: EnvConfig | None = None,
                 perception_config: PerceptionConfig | None = None,
                 embodiment: EmbodimentConfig | None = None):
        self.pcfg = perception_config or PerceptionConfig()
        self._prng = np.random.default_rng(self.pcfg.seed)
        super().__init__(config, embodiment=embodiment or GRIPPER_2F)

        pts_dim = self.pcfg.n_points * 4
        mask_dim = self.pcfg.n_points
        # proprio: ee_xyz(3) + gripper_open(1) + embodiment one-hot(N)
        proprio_dim = 4 + N_EMBODIMENTS
        obs_dim = pts_dim + mask_dim + proprio_dim
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # exposed for building the matching PointCloudStateEncoder
        self.n_points = self.pcfg.n_points
        self.point_channels = 4
        self.proprio_dim = proprio_dim

    # ------------------------------------------------------------------ #
    # Synthetic point-cloud generation
    # ------------------------------------------------------------------ #
    def _sample_cube_surface_points(self, n: int, cube_pos, cube_quat):
        """Sample n points uniformly on the 6 faces of the (rotated) cube."""
        h = self.pcfg.cube_half_extent
        local = self._prng.uniform(-h, h, size=(n, 3))
        face = self._prng.integers(0, 3, size=n)
        sign = self._prng.choice([-1.0, 1.0], size=n)
        for axis in range(3):
            sel = face == axis
            local[sel, axis] = sign[sel] * h
        rot = np.array(p.getMatrixFromQuaternion(cube_quat)).reshape(3, 3)
        world = local @ rot.T + np.array(cube_pos)
        return world.astype(np.float32)

    def _sample_table_points(self, n: int, cube_pos):
        r = self.pcfg.table_sample_radius
        offsets = self._prng.uniform(-r, r, size=(n, 2))
        xy = np.array(cube_pos[:2]) + offsets
        xy[:, 0] = np.clip(xy[:, 0], *sorted([self.cfg.workspace_low[0] - 0.1,
                                               self.cfg.workspace_high[0] + 0.1]))
        z = np.full(n, self.cfg.table_height, dtype=np.float32)
        world = np.concatenate([xy, z[:, None]], axis=1)
        return world.astype(np.float32)

    def _build_point_cloud(self):
        n = self.pcfg.n_points
        n_obj = max(1, int(n * self.pcfg.object_point_frac))
        n_bg = n - n_obj

        cube_pos, cube_quat = p.getBasePositionAndOrientation(self._cube)
        obj_pts = self._sample_cube_surface_points(n_obj, cube_pos, cube_quat)
        bg_pts = self._sample_table_points(n_bg, cube_pos)

        pts = np.concatenate([obj_pts, bg_pts], axis=0)
        is_object = np.concatenate(
            [np.ones(n_obj, dtype=np.float32), np.zeros(n_bg, dtype=np.float32)]
        )
        valid = np.ones(n, dtype=np.float32)

        # --- sensor noise: Gaussian jitter ---
        pts = pts + self._prng.normal(0.0, self.pcfg.jitter_std, size=pts.shape)

        # --- dropout (random missing returns, e.g. low reflectance) ---
        drop_mask = self._prng.random(n) < self.pcfg.dropout_frac
        valid[drop_mask] = 0.0

        # --- occlusion wedge (e.g. arm/gripper blocking part of the scene) ---
        if self._prng.random() < self.pcfg.occlusion_prob:
            center = pts[self._prng.integers(0, n)]
            dists = np.linalg.norm(pts - center, axis=1)
            k = int(n * self.pcfg.occlusion_frac)
            occluded_idx = np.argsort(dists)[:k]
            valid[occluded_idx] = 0.0

        # zero out invalid points/features so padding is a clean, consistent
        # signal for the encoder (mask channel is the real "this is padding"
        # indicator; zeroing the payload avoids leaking noise structure)
        pts[valid == 0] = 0.0
        is_object[valid == 0] = 0.0

        points_4d = np.concatenate([pts, is_object[:, None]], axis=1)
        # shuffle so "object points always come first" isn't a shortcut the
        # policy can exploit instead of actually using the mask/xyz
        perm = self._prng.permutation(n)
        return points_4d[perm].astype(np.float32), valid[perm].astype(np.float32)

    # ------------------------------------------------------------------ #
    # Gym API override
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        # ground-truth obs still computed (parent class) for reward use only
        self._gt_obs = super()._get_obs()

        points, valid = self._build_point_cloud()
        ee_pos, _ = self._get_ee_pose()
        finger_states = [p.getJointState(self._robot, j)[0]
                          for j in self.FINGER_JOINTS]
        gripper_open = float(np.mean(finger_states) > 0.02)
        onehot = np.zeros(N_EMBODIMENTS, dtype=np.float32)
        onehot[self.embodiment.embodiment_index] = 1.0
        proprio = np.concatenate([
            np.array([*ee_pos, gripper_open], dtype=np.float32), onehot
        ])

        obs = np.concatenate([points.flatten(), valid, proprio])
        return obs.astype(np.float32)

    def _compute_reward(self, obs: np.ndarray):
        # reward/success must use privileged ground-truth state, not the
        # noisy point cloud we just handed to the policy
        return super()._compute_reward(self._gt_obs)


def make_perception_env(render: bool = False, seed: int | None = None,
                         perception_config: PerceptionConfig | None = None,
                         embodiment: EmbodimentConfig | None = None
                         ) -> ReachPickPushPerceptionEnv:
    pcfg = perception_config or PerceptionConfig(seed=seed)
    return ReachPickPushPerceptionEnv(EnvConfig(render=render, seed=seed),
                                       perception_config=pcfg,
                                       embodiment=embodiment)


if __name__ == "__main__":
    env = make_perception_env(render=True, seed=0)
    obs, _ = env.reset()
    print("obs shape:", obs.shape, "expected:",
          env.observation_space.shape)
    import time
    for _ in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(1.0 / 60.0)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
