"""
Stage 5 (core experiment): measure ZERO-SHOT success rate of the shared
model on a held-out embodiment it never trained on, vs an embodiment-
specific model trained from scratch on that embodiment.

This is the project's headline claim, so be precise about what it
measures: train the "shared" model on embodiments {A, B} together
(scripts/train_ppo_stage4.py --mode shared), train two "specific" models,
one per embodiment, each seeing ONLY its own embodiment during training.
Then:
  - in-distribution check: specific-A on A, specific-B on B (sanity —
    should both be strong, same as Stage 2)
  - zero-shot transfer: shared model evaluated on A and on B separately,
    vs. specific-A evaluated on B (embodiment it never saw) and
    specific-B evaluated on A. The gap between "shared on held-out" and
    "specific trained on a DIFFERENT embodiment" is the zero-shot
    transfer result. Don't confuse this with an interpolation/average
    score — every number here is a true zero-shot rollout, no fine-tuning.

Usage:
    python scripts/eval_zero_shot_transfer.py \
        --shared checkpoints/ppo_stage4/shared/best_model.zip \
        --specific-2f checkpoints/ppo_stage4/specific_2f_parallel_jaw/best_model.zip \
        --specific-3f checkpoints/ppo_stage4/specific_3f_underactuated/best_model.zip \
        --episodes 50
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.reach_pick_push_perception_env import make_perception_env
from envs.embodiments import get_embodiment


def eval_model(checkpoint, embodiment_id, episodes, seed):
    from utils.checkpoints import load_ppo_checkpoint
    model = load_ppo_checkpoint(checkpoint)
    env = make_perception_env(seed=seed, embodiment=get_embodiment(embodiment_id))
    n_success = 0
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        info = {"success": False}
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            if terminated or truncated:
                break
        n_success += int(info["success"])
        returns.append(ep_return)
    env.close()
    return n_success / episodes, float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=str, required=True)
    parser.add_argument("--specific-2f", type=str, required=True)
    parser.add_argument("--specific-3f", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    for path, label in [(args.shared, "shared"),
                         (args.specific_2f, "specific-2f"),
                         (args.specific_3f, "specific-3f")]:
        if not os.path.exists(path):
            print(f"WARNING: {label} checkpoint not found at {path}")

    results = {}
    results[("shared", "2f_parallel_jaw")] = eval_model(
        args.shared, "2f_parallel_jaw", args.episodes, args.seed)
    results[("shared", "3f_underactuated")] = eval_model(
        args.shared, "3f_underactuated", args.episodes, args.seed)
    results[("specific-2f", "2f_parallel_jaw")] = eval_model(
        args.specific_2f, "2f_parallel_jaw", args.episodes, args.seed)
    results[("specific-2f", "3f_underactuated")] = eval_model(
        args.specific_2f, "3f_underactuated", args.episodes, args.seed)
    results[("specific-3f", "3f_underactuated")] = eval_model(
        args.specific_3f, "3f_underactuated", args.episodes, args.seed)
    results[("specific-3f", "2f_parallel_jaw")] = eval_model(
        args.specific_3f, "2f_parallel_jaw", args.episodes, args.seed)

    print(f"\n=== Zero-shot cross-embodiment transfer ({args.episodes} eps/cell) ===")
    print(f"{'Model':<14} {'Eval embodiment':<20} {'Success':>9} "
          f"{'Return':>10}  Note")
    def row(model, embodiment, note):
        sr, mret = results[(model, embodiment)]
        print(f"{model:<14} {embodiment:<20} {sr:>8.1%} {mret:>10.2f}  {note}")

    row("specific-2f", "2f_parallel_jaw", "in-distribution (sanity)")
    row("specific-3f", "3f_underactuated", "in-distribution (sanity)")
    print("-" * 70)
    row("shared", "2f_parallel_jaw", "shared, seen 2f in training")
    row("shared", "3f_underactuated", "shared, seen 3f in training")
    print("-" * 70)
    row("specific-2f", "3f_underactuated", "ZERO-SHOT: never trained on 3f")
    row("specific-3f", "2f_parallel_jaw", "ZERO-SHOT: never trained on 2f")

    print("\nHeadline comparison — shared model's cross-embodiment "
          "robustness vs an embodiment-specific model asked to generalize:")
    shared_2f = results[("shared", "2f_parallel_jaw")][0]
    shared_3f = results[("shared", "3f_underactuated")][0]
    zs_2f = results[("specific-3f", "2f_parallel_jaw")][0]
    zs_3f = results[("specific-2f", "3f_underactuated")][0]
    print(f"  shared success (avg over both embodiments):     "
          f"{(shared_2f + shared_3f) / 2:.1%}")
    print(f"  embodiment-specific zero-shot (avg, never seen): "
          f"{(zs_2f + zs_3f) / 2:.1%}")


if __name__ == "__main__":
    main()
