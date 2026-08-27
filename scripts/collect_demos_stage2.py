"""
Collect demonstrations for Stage 2/3 noisy-state behavior cloning.

The scripted oracle still ACTS on privileged ground-truth state (that's
fine — an oracle is allowed privileged info, same as Stage 1). What's new
here is that we RECORD the noisy point-cloud observation the perception
env produced at each step, not the GT state, so the resulting dataset
teaches a BC policy to map noisy point clouds -> actions.

Usage:
    python scripts/collect_demos_stage2.py --episodes 200 \
        --out demos/stage2_demos.npz --embodiment 2f_parallel_jaw
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from envs.reach_pick_push_perception_env import make_perception_env
from envs.embodiments import get_embodiment
from policies.scripted_policy import ScriptedPickPushPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out", type=str, default="demos/stage2_demos.npz")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embodiment", type=str, default="2f_parallel_jaw",
                         choices=["2f_parallel_jaw", "3f_underactuated"])
    args = parser.parse_args()

    embodiment = get_embodiment(args.embodiment)
    env = make_perception_env(render=args.render, seed=args.seed,
                               embodiment=embodiment)
    scripted = ScriptedPickPushPolicy()

    all_obs, all_actions, all_ep_ids = [], [], []
    n_success = 0

    pbar = tqdm(range(args.episodes), desc=f"Collecting demos ({embodiment.id})")
    for ep in pbar:
        obs, _ = env.reset(seed=args.seed + ep)
        scripted.reset()
        success = False
        for t in range(args.max_steps):
            # oracle acts on privileged GT state, we record the noisy obs
            action = scripted.act(env._gt_obs)
            all_obs.append(obs.copy())
            all_actions.append(action.copy())
            all_ep_ids.append(ep)

            obs, reward, terminated, truncated, info = env.step(action)
            if info["success"]:
                success = True
            if terminated or truncated:
                break
        if success:
            n_success += 1
        pbar.set_postfix(success_rate=f"{n_success / (ep + 1):.2f}")

    env.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        observations=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        episode_ids=np.array(all_ep_ids, dtype=np.int32),
        embodiment_id=embodiment.id,
        n_points=env.n_points,
        point_channels=env.point_channels,
        proprio_dim=env.proprio_dim,
    )
    print(f"\nSaved {len(all_obs)} transitions from {args.episodes} episodes "
          f"({n_success}/{args.episodes} scripted-policy successes) to {args.out}")
    if n_success / args.episodes < 0.7:
        print("WARNING: scripted policy success rate is low under noisy "
              "perception observation logging (it still acts on GT state, "
              "so a low rate here usually means grasp/goal tolerances for "
              "this embodiment need tuning in envs/embodiments.py).")


if __name__ == "__main__":
    main()
