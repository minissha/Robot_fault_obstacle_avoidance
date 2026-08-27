"""
Train PPO on the Stage 2 noisy-point-cloud task, using PointNetLite as the
SB3 features extractor.

Usage:
    python scripts/train_ppo_stage2.py --config configs/stage2_noisy_perception.yaml
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
from envs.embodiments import get_embodiment, N_EMBODIMENTS
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
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--embodiment", type=str, default="2f_parallel_jaw",
                         choices=["2f_parallel_jaw", "3f_underactuated"])
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
    env = DummyVecEnv([build_env_fn(env_cfg, perc_cfg, seed + i, args.embodiment)
                        for i in range(n_envs)])
    eval_env = DummyVecEnv([build_env_fn(env_cfg, perc_cfg, seed + 1000,
                                          args.embodiment)])

    os.makedirs(ppo_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(ppo_cfg["log_dir"], exist_ok=True)

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
        tensorboard_log=ppo_cfg["log_dir"],
        seed=seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=ppo_cfg["checkpoint_dir"],
        log_path=ppo_cfg["log_dir"],
        eval_freq=max(ppo_cfg["eval_freq"] // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    total_timesteps = args.timesteps or ppo_cfg["total_timesteps"]
    model.learn(total_timesteps=total_timesteps, callback=eval_callback,
                progress_bar=True)

    final_path = os.path.join(ppo_cfg["checkpoint_dir"], "final_model.zip")
    model.save(final_path)
    print(f"\nSaved final model to {final_path}")


if __name__ == "__main__":
    main()
