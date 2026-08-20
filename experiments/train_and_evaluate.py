"""
train_and_evaluate.py

Two responsibilities, selectable via --stage:

  1. "train_ann"  -- load results/dataset.npz (from generate_data.py),
     perform an episode-level 70/15/15 train/val/test split, train a
     small MLPRegressor (controllers.ann_imitator.build_mlp), report
     test-set MAE, and save the model to results/ann_model.joblib.

  2. "evaluate"   -- run the full experimental grid:
         4 controllers x 2 tiers x 2 sensor conditions x 15 seeds = 240 trials
     using identical seeds across controllers for fair paired comparison.
     Results are appended incrementally to results/evaluation_results.csv
     (and mirrored to results/evaluation_results.json) after every trial,
     so the run can be safely interrupted and resumed: already-completed
     (controller, tier, condition, seed) combinations are skipped.

Run stages independently, or "--stage all" to run both in sequence
(requires results/ann_model.joblib to not yet exist, or it will be reused).

Run from the project root, e.g.:
    python experiments/train_and_evaluate.py --stage train_ann
    python experiments/train_and_evaluate.py --stage evaluate
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simulation_core import RobotSimulator
from controllers.baseline import BaselineController
from controllers.fuzzy_handtuned import HandTunedFLC
from controllers.fuzzy_nsga2 import load_optimized_controller
from controllers.ann_imitator import ANNController, build_mlp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

NSGA2_PATH = os.path.join(RESULTS_DIR, "nsga2_result.npz")
DATASET_PATH = os.path.join(RESULTS_DIR, "dataset.npz")
ANN_MODEL_PATH = os.path.join(RESULTS_DIR, "ann_model.joblib")
EVAL_CSV_PATH = os.path.join(RESULTS_DIR, "evaluation_results.csv")
EVAL_JSON_PATH = os.path.join(RESULTS_DIR, "evaluation_results.json")

N_SEEDS = 15
EVAL_SEED_BASE = 555
TIERS = ("sparse", "dense")
CONDITIONS = (False, True)  # faulty flags: clean, faulty

METRIC_FIELDS = [
    "success", "collision", "path_length", "time_to_goal",
    "min_clearance", "steering_smoothness", "final_goal_distance", "steps",
]
CSV_FIELDS = ["controller", "tier", "condition", "seed", "trial_seed"] + METRIC_FIELDS


# --------------------------------------------------------------------------
# Stage 1: train ANN
# --------------------------------------------------------------------------

def episode_split(episode_ids: np.ndarray, seed: int = 0):
    """Split unique episode IDs into 70/15/15 train/val/test sets."""
    unique_eps = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_eps)

    n = len(shuffled)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))

    train_eps = set(shuffled[:n_train].tolist())
    val_eps = set(shuffled[n_train:n_train + n_val].tolist())
    test_eps = set(shuffled[n_train + n_val:].tolist())

    return train_eps, val_eps, test_eps


def train_ann(seed: int = 0) -> None:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run experiments/generate_data.py first."
        )

    data = np.load(DATASET_PATH)
    X, y, episode_ids = data["X"], data["y"], data["episode_ids"]

    train_eps, val_eps, test_eps = episode_split(episode_ids, seed=seed)

    def mask_for(eps):
        return np.isin(episode_ids, list(eps))

    train_mask, val_mask, test_mask = mask_for(train_eps), mask_for(val_eps), mask_for(test_eps)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Episode split: {len(train_eps)} train / {len(val_eps)} val / {len(test_eps)} test episodes")
    print(f"Sample split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test rows")

    # scikit-learn's MLPRegressor manages its own internal validation split
    # for early stopping; we combine train+val here and hold test out purely
    # for the final reported MAE, per the blueprint's 70/15/15 episode split.
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    model = build_mlp(seed=seed)
    model.fit(X_fit, y_fit)

    y_pred = model.predict(X_test)
    mae = float(np.mean(np.abs(y_pred - y_test)))
    print(f"Test-set MAE: {mae:.4f} degrees (n_test={len(X_test)})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    controller = ANNController(model=model)
    controller.save(ANN_MODEL_PATH)
    print(f"Saved trained ANN to {ANN_MODEL_PATH}")

    with open(os.path.join(RESULTS_DIR, "ann_training_report.json"), "w") as f:
        json.dump({
            "test_mae_degrees": mae,
            "n_train_episodes": len(train_eps),
            "n_val_episodes": len(val_eps),
            "n_test_episodes": len(test_eps),
            "n_train_rows": int(len(X_fit)),
            "n_test_rows": int(len(X_test)),
        }, f, indent=2)


# --------------------------------------------------------------------------
# Stage 2: full 240-trial evaluation
# --------------------------------------------------------------------------

def build_controllers() -> dict:
    controllers = {"A_baseline": BaselineController(), "B_handtuned_flc": HandTunedFLC()}

    if os.path.exists(NSGA2_PATH):
        controllers["D_nsga2_flc"] = load_optimized_controller(NSGA2_PATH)
    else:
        print(f"WARNING: {NSGA2_PATH} not found; skipping D_nsga2_flc. "
              f"Run the NSGA-II optimization stage first.")

    if os.path.exists(ANN_MODEL_PATH):
        controllers["E_ann_imitator"] = ANNController.load(ANN_MODEL_PATH)
    else:
        print(f"WARNING: {ANN_MODEL_PATH} not found; skipping E_ann_imitator. "
              f"Run --stage train_ann first.")

    return controllers


def load_completed_trials() -> set:
    """Read the CSV of already-completed trials (if any) for resume support."""
    completed = set()
    if not os.path.exists(EVAL_CSV_PATH):
        return completed
    with open(EVAL_CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["controller"], row["tier"], row["condition"], int(row["seed"]))
            completed.add(key)
    return completed


def append_trial_row(row: dict) -> None:
    file_exists = os.path.exists(EVAL_CSV_PATH)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(EVAL_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def rewrite_json_mirror() -> None:
    """Rebuild the JSON mirror of all completed trials from the CSV."""
    rows = []
    if os.path.exists(EVAL_CSV_PATH):
        with open(EVAL_CSV_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    with open(EVAL_JSON_PATH, "w") as f:
        json.dump(rows, f, indent=2)


def run_evaluation() -> None:
    controllers = build_controllers()
    if not controllers:
        raise RuntimeError("No controllers available to evaluate.")

    completed = load_completed_trials()

    base_rng = np.random.default_rng(EVAL_SEED_BASE)
    trial_seeds = [int(s) for s in base_rng.integers(0, 1_000_000, size=N_SEEDS)]

    total_trials = len(controllers) * len(TIERS) * len(CONDITIONS) * N_SEEDS
    done = 0

    for controller_name, controller in controllers.items():
        for tier in TIERS:
            for faulty in CONDITIONS:
                condition = "faulty" if faulty else "clean"
                for seed_idx, trial_seed in enumerate(trial_seeds):
                    key = (controller_name, tier, condition, seed_idx)
                    if key in completed:
                        done += 1
                        continue

                    sim = RobotSimulator(tier=tier, seed=trial_seed, faulty=faulty)
                    result = sim.run(controller)
                    m = result.metrics.to_dict()

                    row = {
                        "controller": controller_name,
                        "tier": tier,
                        "condition": condition,
                        "seed": seed_idx,
                        "trial_seed": trial_seed,
                        **m,
                    }
                    append_trial_row(row)
                    done += 1
                    print(f"[{done}/{total_trials}] {controller_name} | {tier} | {condition} | "
                          f"seed_idx={seed_idx} -> success={m['success']}, collision={m['collision']}")

    rewrite_json_mirror()
    print(f"\nEvaluation complete: {done}/{total_trials} trials recorded in {EVAL_CSV_PATH}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ANN and/or run the full evaluation grid.")
    parser.add_argument(
        "--stage", choices=["train_ann", "evaluate", "all"], default="all",
        help="Which stage to run.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the ANN train/val/test split.")
    args = parser.parse_args()

    if args.stage in ("train_ann", "all"):
        train_ann(seed=args.seed)
    if args.stage in ("evaluate", "all"):
        run_evaluation()


if __name__ == "__main__":
    main()
