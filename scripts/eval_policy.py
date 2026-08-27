"""
Evaluate a policy (random / scripted-oracle / ppo / bc) on the Stage 1 task
and report success rate.

Usage:
    python scripts/eval_policy.py --random --episodes 3 --render
    python scripts/eval_policy.py --scripted --episodes 20
    python scripts/eval_policy.py --checkpoint checkpoints/ppo_stage1/best_model.zip --policy-type ppo --episodes 50
    python scripts/eval_policy.py --checkpoint checkpoints/bc_stage1.pt --policy-type bc --episodes 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.reach_pick_push_env import make_env
from envs.reach_pick_push_perception_env import make_perception_env
from envs.embodiments import get_embodiment
from policies.scripted_policy import ScriptedPickPushPolicy


def load_policy(policy_type: str, checkpoint: str | None):
    if policy_type == "random":
        return None  # handled inline
    if policy_type == "scripted":
        return ScriptedPickPushPolicy()
    if policy_type == "ppo":
        from utils.checkpoints import load_ppo_checkpoint
        model = load_ppo_checkpoint(checkpoint)
        return model
    if policy_type == "bc":
        import torch
        from policies.bc_policy import BCPolicy
        model = BCPolicy()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.eval()
        return model
    if policy_type == "bc_stage2":
        import torch
        from policies.bc_policy import PointCloudBCPolicy
        ckpt = torch.load(checkpoint, map_location="cpu")
        model = PointCloudBCPolicy(
            n_points=ckpt["n_points"], point_channels=ckpt["point_channels"],
            proprio_dim=ckpt["proprio_dim"], embed_dim=ckpt["embed_dim"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
    raise ValueError(f"Unknown policy_type: {policy_type}")


def get_action(policy_type, policy, obs, env):
    if policy_type == "random":
        return env.action_space.sample()
    if policy_type == "scripted":
        return policy.act(obs)
    if policy_type == "ppo":
        action, _ = policy.predict(obs, deterministic=True)
        return action
    if policy_type in ("bc", "bc_stage2"):
        return policy.act(obs)
    raise ValueError(policy_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--policy-type", type=str, default=None,
                         choices=["ppo", "bc", "bc_stage2"])
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stage2", action="store_true",
                         help="use the noisy point-cloud perception env "
                              "instead of ground-truth state")
    parser.add_argument("--embodiment", type=str, default="2f_parallel_jaw",
                         choices=["2f_parallel_jaw", "3f_underactuated"],
                         help="only used with --stage2")
    args = parser.parse_args()

    if args.random:
        policy_type = "random"
    elif args.scripted:
        policy_type = "scripted"
    else:
        assert args.checkpoint and args.policy_type, (
            "Provide --checkpoint and --policy-type, or use --random / --scripted"
        )
        policy_type = args.policy_type

    policy = load_policy(policy_type, args.checkpoint)
    env = (make_perception_env(render=args.render, seed=args.seed,
                                embodiment=get_embodiment(args.embodiment))
           if args.stage2 else make_env(render=args.render, seed=args.seed))

    n_success = 0
    returns = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if policy_type == "scripted":
            policy.reset()
        ep_return = 0.0
        for t in range(args.max_steps):
            action = get_action(policy_type, policy, obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            if args.render:
                time.sleep(1.0 / 60.0)
            if terminated or truncated:
                break
        if info["success"]:
            n_success += 1
        returns.append(ep_return)
        print(f"episode {ep:3d} | return {ep_return:7.2f} | "
              f"success {info['success']} | steps {t + 1}")

    env.close()

    success_rate = n_success / args.episodes
    print(f"\n=== {policy_type} over {args.episodes} episodes ===")
    print(f"success rate: {success_rate:.2%}")
    print(f"mean return:  {np.mean(returns):.2f} +/- {np.std(returns):.2f}")


if __name__ == "__main__":
    main()
