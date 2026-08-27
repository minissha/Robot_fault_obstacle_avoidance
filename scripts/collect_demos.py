"""
Collect demonstrations using the hand-scripted oracle policy, for later
behavior cloning.

Usage:
    python scripts/collect_demos.py --episodes 200 --out demos/stage1_demos.npz
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from envs.reach_pick_push_env import make_env
from policies.scripted_policy import ScriptedPickPushPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out", type=str, default="demos/stage1_demos.npz")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = make_env(render=args.render, seed=args.seed)
    scripted = ScriptedPickPushPolicy()

    all_obs, all_actions, all_ep_ids = [], [], []
    n_success = 0

    pbar = tqdm(range(args.episodes), desc="Collecting demos")
    for ep in pbar:
        obs, _ = env.reset(seed=args.seed + ep)
        scripted.reset()
        success = False
        for t in range(args.max_steps):
            action = scripted.act(obs)
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
    )
    print(f"\nSaved {len(all_obs)} transitions from {args.episodes} episodes "
          f"({n_success}/{args.episodes} scripted-policy successes) to {args.out}")
    if n_success / args.episodes < 0.7:
        print("WARNING: scripted policy success rate is low. BC will learn "
              "from noisy/failed demos. Consider tuning ScriptedPickPushPolicy "
              "thresholds in policies/scripted_policy.py before collecting more.")


if __name__ == "__main__":
    main()
