"""
generate_data.py

Builds the imitation-learning dataset for Controller E by watching the
NSGA-II controller (D) drive, and logging what it saw against what it did.

Not just the teacher's own routes, though. A network trained purely on
those only ever sees states a competent driver reaches, so the moment it
makes a small mistake it is somewhere it has no advice for, and the errors
compound until it hits something. The previous version of this file did
exactly that and produced a controller that matched the teacher closely on
paper and crashed in every episode.

So after the first round the student drives and the teacher is asked what
it would have done at each state the student actually reached, and those
answers get added to the pile (Ross, Gordon and Bagnell, 2011 -- DAgger).
The student ends up with advice covering the situations it gets itself
into, not only the ones the teacher would have.

Requires results/nsga2_result.npz (produced by running
controllers.fuzzy_nsga2.optimize; see train_and_evaluate.py's `--stage
nsga2`, or the ensure_nsga2_result helper below).

Run from the project root:
    python experiments/generate_data.py

Outputs (all under results/):
    X_data.npy            float64, shape (N, OBSERVATION_DIM)
    y_data.npy             float64, shape (N, 2)  -- [steering_deg, speed]
    episode_ids.npy        int64,   shape (N,)
    episode_metadata.json  per-episode tier/seed/outcome info
    dataset.npz             bundle of the above
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simulation_core import RobotSimulator, build_full_observation, unpack_action
from controllers.ann_imitator import ANNController, build_mlp
from controllers.fuzzy_nsga2 import load_optimized_controller, optimize

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
NSGA2_PATH = os.path.join(RESULTS_DIR, "nsga2_result.npz")

N_EPISODES = 80          # teacher episodes in the first round
DAGGER_ROUNDS = 3        # extra rounds where the student drives
DAGGER_EPISODES = 40     # episodes per extra round
DATA_SEED = 2026         # base seed for episode seed generation


def ensure_nsga2_result() -> None:
    """Run NSGA-II and save the result if results/nsga2_result.npz is missing."""
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


def _teacher_labels(teacher, observations: np.ndarray) -> np.ndarray:
    """What the teacher would do at each of these states.

    The teacher is reset first and fed the states in the order they
    happened, because it carries a little state of its own (the cycle
    breaker) and asking it out of order would give answers it never would
    have given in the moment.
    """
    teacher.reset()
    out = np.zeros((len(observations), 2), dtype=np.float64)
    for i, obs in enumerate(observations):
        out[i] = unpack_action(teacher.predict(obs))
    return out


def _rollout(controller, tier: str, seed: int):
    """One episode. Returns (observations, metrics)."""
    sim = RobotSimulator(tier=tier, seed=seed, faulty=False)
    result = sim.run(controller)
    obs = np.array(result.full_observation_log, dtype=np.float64)
    return obs, result.metrics


def generate_dataset(n_episodes: int = N_EPISODES, seed: int = DATA_SEED,
                      dagger_rounds: int = DAGGER_ROUNDS,
                      dagger_episodes: int = DAGGER_EPISODES) -> None:
    ensure_nsga2_result()
    teacher = load_optimized_controller(NSGA2_PATH, which="knee")

    from config import get_eval_trial_seeds, sample_disjoint_seeds
    rng = np.random.default_rng(seed)
    eval_seeds = set(get_eval_trial_seeds())
    total_eps = n_episodes + dagger_rounds * dagger_episodes
    episode_seeds = sample_disjoint_seeds(rng, total_eps, exclude=eval_seeds)

    X_rows, y_rows, ep_rows, metadata = [], [], [], []
    ep_counter = 0

    def collect(driver, seeds, round_label):
        nonlocal ep_counter
        for ep_seed in seeds:
            tier = "sparse" if ep_counter % 2 == 0 else "dense"
            obs, metrics = _rollout(driver, tier, ep_seed)
            if len(obs) == 0:
                continue
            actions = _teacher_labels(teacher, obs)
            X_rows.append(obs)
            y_rows.append(actions)
            ep_rows.append(np.full(len(obs), ep_counter, dtype=np.int64))
            metadata.append({
                "episode_id": ep_counter,
                "round": round_label,
                "driver": "teacher" if driver is teacher else "student",
                "tier": tier,
                "seed": int(ep_seed),
                "success": bool(metrics.success),
                "collision": bool(metrics.collision),
                "n_steps": int(len(obs)),
            })
            ep_counter += 1

    cursor = 0
    collect(teacher, episode_seeds[cursor:cursor + n_episodes], "teacher")
    cursor += n_episodes
    print(f"Round 0 (teacher driving): {len(np.concatenate(X_rows))} samples "
          f"from {ep_counter} episodes")

    for round_idx in range(1, dagger_rounds + 1):
        X = np.vstack(X_rows)
        y = np.vstack(y_rows)
        student_model = build_mlp(seed=round_idx)
        student_model.fit(X, y)
        student = ANNController(model=student_model)

        before = ep_counter
        collect(student, episode_seeds[cursor:cursor + dagger_episodes], f"dagger{round_idx}")
        cursor += dagger_episodes
        reached = sum(1 for m in metadata[before:] if m["success"])
        print(f"Round {round_idx} (student driving, teacher labelling): "
              f"{len(np.vstack(X_rows))} samples total, student reached the goal "
              f"in {reached}/{ep_counter - before} of its own episodes")

    X = np.vstack(X_rows)
    y = np.vstack(y_rows)
    episode_ids = np.concatenate(ep_rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, "X_data.npy"), X)
    np.save(os.path.join(RESULTS_DIR, "y_data.npy"), y)
    np.save(os.path.join(RESULTS_DIR, "episode_ids.npy"), episode_ids)
    with open(os.path.join(RESULTS_DIR, "episode_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    np.savez(os.path.join(RESULTS_DIR, "dataset.npz"), X=X, y=y, episode_ids=episode_ids)

    n_success = sum(1 for m in metadata if m["success"])
    print(f"\nGenerated {len(X)} samples from {len(metadata)} episodes "
          f"({n_success} reached the goal). X {X.shape}, y {y.shape}.")
    print(f"Saved to {RESULTS_DIR}/ (X_data.npy, y_data.npy, episode_ids.npy, "
          f"episode_metadata.json, dataset.npz)")


if __name__ == "__main__":
    generate_dataset()
