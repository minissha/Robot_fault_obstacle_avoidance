"""
generate_data.py

Generates a reproducible imitation-learning dataset by running the
NSGA-II-optimized FLC (Controller D) across randomized clean-condition
episodes, logging (front, left, right) -> steering pairs at every
timestep.

Requires results/nsga2_result.npz to already exist (produced by
running controllers/fuzzy_nsga2.optimize and saving its output; see
train_and_evaluate.py's `--stage nsga2` step, or run this module's
`ensure_nsga2_result` helper).

Run from the project root:
    python experiments/generate_data.py

Outputs (all under results/):
    X_data.npy            float64 array, shape (N, 3)   -- sensor readings
    y_data.npy             float64 array, shape (N,)      -- steering targets
    episode_ids.npy        int64 array, shape (N,)         -- episode index per sample
    episode_metadata.json  per-episode tier/seed/success/step-count info
    dataset.npz             convenience bundle of the above four arrays
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simulation_core import RobotSimulator
from controllers.fuzzy_nsga2 import load_optimized_controller, optimize

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
NSGA2_PATH = os.path.join(RESULTS_DIR, "nsga2_result.npz")

N_EPISODES = 80          # within the blueprint's 60-100 range
DATA_SEED = 2026           # base seed for episode seed generation


def ensure_nsga2_result() -> None:
    """Run NSGA-II and save its result if results/nsga2_result.npz doesn't exist yet."""
    if os.path.exists(NSGA2_PATH):
        return
    print("No existing NSGA-II result found -- running optimization now "
          "(this can take a few minutes)...")
    result = optimize(seed=0, verbose=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(
        NSGA2_PATH,
        pareto_X=result["pareto_X"],
        pareto_F=result["pareto_F"],
        knee_index=result["knee_index"],
        knee_genes=result["knee_genes"],
        safety_index=result["safety_index"],
        safety_genes=result["safety_genes"],
        efficiency_index=result["efficiency_index"],
        efficiency_genes=result["efficiency_genes"],
        training_seeds=result["training_seeds"],
        convergence_hypervolume=result["convergence_hypervolume"],
        convergence_gen_F_min=result["convergence_gen_F_min"],
        convergence_gen_F_mean=result["convergence_gen_F_mean"],
    )
    print(f"Saved NSGA-II result to {NSGA2_PATH}")


def generate_dataset(n_episodes: int = N_EPISODES, seed: int = DATA_SEED) -> None:
    ensure_nsga2_result()
    controller = load_optimized_controller(NSGA2_PATH, which="knee")

    from config import get_eval_trial_seeds, sample_disjoint_seeds
    rng = np.random.default_rng(seed)
    eval_seeds = set(get_eval_trial_seeds())
    episode_seeds = sample_disjoint_seeds(rng, n_episodes, exclude=eval_seeds)

    X_rows = []
    y_rows = []
    episode_id_rows = []
    metadata = []

    for ep_idx, ep_seed in enumerate(episode_seeds):
        tier = "sparse" if ep_idx % 2 == 0 else "dense"
        sim = RobotSimulator(tier=tier, seed=ep_seed, faulty=False)
        result = sim.run(controller)

        n_steps = len(result.steering_log)
        for step in range(n_steps):
            X_rows.append(result.sensor_log[step])
            y_rows.append(result.steering_log[step])
            episode_id_rows.append(ep_idx)

        metadata.append({
            "episode_id": ep_idx,
            "tier": tier,
            "seed": ep_seed,
            "success": bool(result.metrics.success),
            "collision": bool(result.metrics.collision),
            "n_steps": n_steps,
        })

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.float64)
    episode_ids = np.array(episode_id_rows, dtype=np.int64)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, "X_data.npy"), X)
    np.save(os.path.join(RESULTS_DIR, "y_data.npy"), y)
    np.save(os.path.join(RESULTS_DIR, "episode_ids.npy"), episode_ids)

    with open(os.path.join(RESULTS_DIR, "episode_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    np.savez(
        os.path.join(RESULTS_DIR, "dataset.npz"),
        X=X, y=y, episode_ids=episode_ids,
    )

    n_success = sum(1 for m in metadata if m["success"])
    print(f"Generated {len(X)} samples from {n_episodes} episodes "
          f"({n_success}/{n_episodes} reached the goal).")
    print(f"Saved to {RESULTS_DIR}/ (X_data.npy, y_data.npy, episode_ids.npy, "
          f"episode_metadata.json, dataset.npz)")


if __name__ == "__main__":
    generate_dataset()
