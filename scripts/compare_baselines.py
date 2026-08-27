"""
Stage 3: run the three baselines your project plan calls for and print a
single comparison table:
  1. GT-state PPO       (Stage 1, upper bound)
  2. noisy-state PPO    (Stage 2)
  3. noisy-state BC     (Stage 2 demos -> PointCloudBCPolicy)

Usage:
    python scripts/compare_baselines.py \
        --gt-ppo checkpoints/ppo_stage1/best_model.zip \
        --noisy-ppo checkpoints/ppo_stage2/best_model.zip \
        --noisy-bc checkpoints/bc_stage2.pt \
        --episodes 50
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.reach_pick_push_env import make_env
from envs.reach_pick_push_perception_env import make_perception_env
from envs.embodiments import get_embodiment


def run_gt_ppo(checkpoint, episodes, seed, embodiment_id):
    from utils.checkpoints import load_ppo_checkpoint
    model = load_ppo_checkpoint(checkpoint)
    env = make_env(seed=seed, embodiment=get_embodiment(embodiment_id))
    return _rollout(env, episodes, seed,
                     lambda obs: model.predict(obs, deterministic=True)[0])


def run_noisy_ppo(checkpoint, episodes, seed, embodiment_id):
    from utils.checkpoints import load_ppo_checkpoint
    model = load_ppo_checkpoint(checkpoint)
    env = make_perception_env(seed=seed, embodiment=get_embodiment(embodiment_id))
    return _rollout(env, episodes, seed,
                     lambda obs: model.predict(obs, deterministic=True)[0])


def run_noisy_bc(checkpoint, episodes, seed, embodiment_id):
    import torch
    from policies.bc_policy import PointCloudBCPolicy
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = PointCloudBCPolicy(
        n_points=ckpt["n_points"], point_channels=ckpt["point_channels"],
        proprio_dim=ckpt["proprio_dim"], embed_dim=ckpt["embed_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    env = make_perception_env(seed=seed, embodiment=get_embodiment(embodiment_id))
    return _rollout(env, episodes, seed, lambda obs: model.act(obs))


def _rollout(env, episodes, seed, action_fn, max_steps=200):
    n_success = 0
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        info = {"success": False}
        for _ in range(max_steps):
            action = action_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            if terminated or truncated:
                break
        n_success += int(info["success"])
        returns.append(ep_return)
    env.close()
    return n_success / episodes, float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-ppo", type=str, default=None,
                         help="checkpoints/ppo_stage1/best_model.zip")
    parser.add_argument("--noisy-ppo", type=str, default=None,
                         help="checkpoints/ppo_stage2/best_model.zip")
    parser.add_argument("--noisy-bc", type=str, default=None,
                         help="checkpoints/bc_stage2.pt")
    parser.add_argument("--embodiment", type=str, default="2f_parallel_jaw",
                         choices=["2f_parallel_jaw", "3f_underactuated"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows = []
    if args.gt_ppo and os.path.exists(args.gt_ppo):
        sr, mret, sret = run_gt_ppo(args.gt_ppo, args.episodes, args.seed,
                                     args.embodiment)
        rows.append(("GT-state PPO (Stage 1)", sr, mret, sret))
    else:
        print("skipping GT-state PPO: no --gt-ppo checkpoint given/found")

    if args.noisy_ppo and os.path.exists(args.noisy_ppo):
        sr, mret, sret = run_noisy_ppo(args.noisy_ppo, args.episodes, args.seed,
                                        args.embodiment)
        rows.append(("Noisy-state PPO (Stage 2)", sr, mret, sret))
    else:
        print("skipping noisy-state PPO: no --noisy-ppo checkpoint given/found")

    if args.noisy_bc and os.path.exists(args.noisy_bc):
        sr, mret, sret = run_noisy_bc(args.noisy_bc, args.episodes, args.seed,
                                       args.embodiment)
        rows.append(("Noisy-state BC (Stage 2)", sr, mret, sret))
    else:
        print("skipping noisy-state BC: no --noisy-bc checkpoint given/found")

    if not rows:
        print("Nothing to compare — pass at least one checkpoint.")
        return

    print(f"\n=== Baseline comparison ({args.episodes} episodes, "
          f"embodiment={args.embodiment}) ===")
    print(f"{'Policy':<28} {'Success Rate':>14} {'Mean Return':>16}")
    for name, sr, mret, sret in rows:
        print(f"{name:<28} {sr:>13.1%} {mret:>10.2f} +/- {sret:.2f}")


if __name__ == "__main__":
    main()
