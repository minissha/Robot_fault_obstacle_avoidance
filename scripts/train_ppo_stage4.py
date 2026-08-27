"""
Stage 4: train either
  --mode shared   a single policy+encoder on envs mixed across BOTH
                   embodiments (2-finger and 3-finger), embodiment identity
                   given only via the one-hot in proprio
  --mode specific a policy+encoder trained on ONE embodiment only
                   (baseline for the Stage 5 zero-shot transfer comparison)

Usage:
    python scripts/train_ppo_stage4.py --config configs/stage4_shared_encoder.yaml --mode shared
    python scripts/train_ppo_stage4.py --config configs/stage4_shared_encoder.yaml --mode specific --embodiment 2f_parallel_jaw
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.reach_pick_push_env import EnvConfig
from envs.reach_pick_push_perception_env import (
    ReachPickPushPerceptionEnv, PerceptionConfig,
)
from envs.embodiments import get_embodiment, ALL_EMBODIMENTS, N_EMBODIMENTS
from policies.sb3_pointnet_extractor import PointCloudFeaturesExtractor
from utils.seeding import set_global_seed


def build_env_fn(env_cfg: dict, perc_cfg: dict, seed: int, embodiment_id: str):
    def _init():
        cfg = EnvConfig(render=False, seed=seed, **env_cfg)
        pcfg = PerceptionConfig(seed=seed, **perc_cfg)
        env = ReachPickPushPerceptionEnv(
            cfg, perception_config=pcfg, embodiment=get_embodiment(embodiment_id)
        )
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                         choices=["shared", "specific"])
    parser.add_argument("--embodiment", type=str, default="2f_parallel_jaw",
                         choices=list(ALL_EMBODIMENTS),
                         help="only used with --mode specific")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    seed = env_cfg.pop("seed", 0)
    perc_cfg = dict(cfg["perception"])
    ppo_cfg = cfg["ppo"]
    embed_dim = cfg["encoder"]["embed_dim"]

    set_global_seed(seed)

    n_envs = ppo_cfg["n_envs"]
    if args.mode == "shared":
        # round-robin embodiment assignment across parallel envs so every
        # update batch contains a mix of both embodiments
        embodiment_ids = list(ALL_EMBODIMENTS)
        env_fns = [
            build_env_fn(env_cfg, perc_cfg, seed + i,
                         embodiment_ids[i % len(embodiment_ids)])
            for i in range(n_envs)
        ]
        eval_fns = [
            build_env_fn(env_cfg, perc_cfg, seed + 1000 + j, eid)
            for j, eid in enumerate(embodiment_ids)
        ]
        run_tag = "shared"
    else:
        env_fns = [build_env_fn(env_cfg, perc_cfg, seed + i, args.embodiment)
                   for i in range(n_envs)]
        eval_fns = [build_env_fn(env_cfg, perc_cfg, seed + 1000, args.embodiment)]
        run_tag = f"specific_{args.embodiment}"

    env = DummyVecEnv(env_fns)
    eval_env = DummyVecEnv(eval_fns)

    checkpoint_dir = os.path.join(ppo_cfg["checkpoint_dir"], run_tag)
    log_dir = os.path.join(ppo_cfg["log_dir"], run_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=PointCloudFeaturesExtractor,
        features_extractor_kwargs=dict(
            n_points=perc_cfg["n_points"], point_channels=4,
            proprio_dim=4 + N_EMBODIMENTS, embed_dim=embed_dim,
        ),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=max(ppo_cfg["eval_freq"] // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    total_timesteps = args.timesteps or ppo_cfg["total_timesteps"]
    model.learn(total_timesteps=total_timesteps, callback=eval_callback,
                progress_bar=True)

    final_path = os.path.join(checkpoint_dir, "final_model.zip")
    model.save(final_path)
    print(f"\n[{run_tag}] saved final model to {final_path}")
    print(f"[{run_tag}] best model (by eval) at "
          f"{os.path.join(checkpoint_dir, 'best_model.zip')}")


if __name__ == "__main__":
    main()
