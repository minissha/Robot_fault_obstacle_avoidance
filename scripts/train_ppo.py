"""
Train PPO on the Stage 1 ground-truth-state reach/pick/push task.

Usage:
    python scripts/train_ppo.py --config configs/stage1_reach_pick_push.yaml
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

from envs.reach_pick_push_env import ReachPickPushEnv, EnvConfig
from utils.seeding import set_global_seed


def build_env_fn(env_cfg: dict, seed: int):
    def _init():
        cfg = EnvConfig(render=False, seed=seed, **env_cfg)
        env = ReachPickPushEnv(cfg)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=None,
                         help="override total_timesteps from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = dict(cfg["env"])
    seed = env_cfg.pop("seed", 0)
    ppo_cfg = cfg["ppo"]

    set_global_seed(seed)

    n_envs = ppo_cfg["n_envs"]
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv(
        [build_env_fn(env_cfg, seed + i) for i in range(n_envs)]
    )
    eval_env = DummyVecEnv([build_env_fn(env_cfg, seed + 1000)])

    os.makedirs(ppo_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(ppo_cfg["log_dir"], exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
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
    print(f"Best model (by eval) saved to "
          f"{os.path.join(ppo_cfg['checkpoint_dir'], 'best_model.zip')}")


if __name__ == "__main__":
    main()
