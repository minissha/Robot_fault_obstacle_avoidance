"""
run_full_experiment.py

Blueprint v2 Sections 2.3/2.4/5.1/5.2/5.3/6.2: the upgraded experimental
grid. Supersedes (does not delete) the original train_and_evaluate.py
--stage evaluate grid; this script is what should be used to produce the
paper's final results table.

For each controller in {A_baseline, B_handtuned_flc, D_nsga2_flc (if
results/nsga2_result.npz exists), E_ann_imitator (if
results/ann_model.joblib exists)}, and the single best-clean-performing
controller additionally wrapped as `<name>_fault_aware` (Sec 6.2's 5th
condition, requires results/fault_detector.joblib):

    for tier in (sparse, dense):
        for condition in (clean,
                           fault_dropout, fault_bias, fault_stale,
                           fault_noise_spike_sev1, _sev2, _sev3):
            for seed_idx in range(N_SEEDS):   # N_SEEDS >= 30, config.py

Logs one row per trial to results/full_trial_results.csv with: success,
collision, timeout, stuck_oscillation (failure-mode taxonomy, Sec 5.4/
Figure 8), path_length, min_clearance, steering_smoothness,
final_goal_distance, steps, controller, tier, condition, fault_type,
fault_severity, seed. Resumable (skips already-completed rows).

Run from the project root:
    python experiments/run_full_experiment.py
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import (
    ANN_MODEL_PATH, CONTROLLER_NAMES, FAULT_AWARE_SUFFIX,
    FAULT_DETECTOR_PATH, FULL_TRIAL_CSV, N_SEEDS, NOISE_SPIKE_SEVERITIES,
    NSGA2_PATH, RESULTS_DIR, TIERS, get_eval_trial_seeds,
)
from simulation_core import RobotSimulator, sample_fault_spec_typed
from controllers.baseline import BaselineController
from controllers.fuzzy_handtuned import HandTunedFLC
from controllers.fault_detector import FaultDetector
from controllers.fault_aware import FaultAwareController

STUCK_DISPLACEMENT_THRESHOLD = 15.0  # world units; see classify_failure_mode

METRIC_FIELDS = [
    "success", "collision", "timeout", "stuck_oscillation",
    "path_length", "time_to_goal", "min_clearance",
    "steering_smoothness", "final_goal_distance", "steps",
]
CSV_FIELDS = ["controller", "tier", "condition", "fault_type", "fault_severity",
              "seed_idx", "trial_seed"] + METRIC_FIELDS


def build_conditions():
    """Return list of (condition_name, fault_type_or_None, severity_range_or_None)."""
    conditions = [("clean", None, None)]
    for ft in ("dropout", "bias", "stale"):
        conditions.append((f"fault_{ft}", ft, None))
    for sev, rng in NOISE_SPIKE_SEVERITIES.items():
        conditions.append((f"fault_noise_spike_sev{sev}", "noise_spike", rng))
    return conditions


def classify_failure_mode(result) -> str:
    """
    Failure-mode taxonomy (Blueprint v2 Sec 3 Figure 8, Sec 5.4):
    collision / timeout / stuck_oscillation. A run that timed out (no
    collision, no success) is further split into a genuine "stuck in a
    small region" oscillation vs. a "ran out of time while still making
    progress" timeout, using the bounding box of the second half of the
    trajectory as a simple, inspectable heuristic.
    """
    m = result.metrics
    if m.success:
        return "success"
    if m.collision:
        return "collision"
    traj = np.array(result.trajectory)
    half = traj[len(traj) // 2:]
    if len(half) >= 2:
        span = float(np.hypot(half[:, 0].max() - half[:, 0].min(),
                               half[:, 1].max() - half[:, 1].min()))
        if span < STUCK_DISPLACEMENT_THRESHOLD:
            return "stuck_oscillation"
    return "timeout"


def build_controllers() -> dict:
    controllers = {"A_baseline": BaselineController(), "B_handtuned_flc": HandTunedFLC()}

    if os.path.exists(NSGA2_PATH):
        from controllers.fuzzy_nsga2 import load_optimized_controller
        controllers["D_nsga2_flc"] = load_optimized_controller(NSGA2_PATH, which="knee")
        controllers["D_nsga2_flc_safety"] = load_optimized_controller(NSGA2_PATH, which="safety")
        controllers["D_nsga2_flc_efficiency"] = load_optimized_controller(NSGA2_PATH, which="efficiency")
    else:
        print(f"[skip] {NSGA2_PATH} not found -- run controllers/fuzzy_nsga2.optimize() first "
              f"(requires pymoo).")

    if os.path.exists(ANN_MODEL_PATH):
        from controllers.ann_imitator import ANNController
        controllers["E_ann_imitator"] = ANNController.load(ANN_MODEL_PATH)
    else:
        print(f"[skip] {ANN_MODEL_PATH} not found -- run "
              f"experiments/train_and_evaluate.py --stage train_ann first.")

    return controllers


def add_fault_aware_condition(controllers: dict) -> dict:
    """Sec 6.2's 5th condition: wrap ONE controller (a fixed, documented
    choice -- the NSGA-II FLC if available, else the hand-tuned FLC, matching
    the blueprint's 'best-performing controller' guidance without requiring
    a second full grid just to rank candidates) as fault-aware, so it can be
    compared 1:1 against its own fault-blind row."""
    if not os.path.exists(FAULT_DETECTOR_PATH):
        print(f"[skip] {FAULT_DETECTOR_PATH} not found -- run "
              f"experiments/train_fault_detector.py first.")
        return controllers

    best_name = "D_nsga2_flc" if "D_nsga2_flc" in controllers else "B_handtuned_flc"
    detector = FaultDetector.load(FAULT_DETECTOR_PATH)
    base = controllers[best_name]
    controllers[best_name + FAULT_AWARE_SUFFIX] = FaultAwareController(base, detector)
    print(f"Added fault-aware condition wrapping '{best_name}'.")
    return controllers


def load_completed() -> set:
    completed = set()
    if not os.path.exists(FULL_TRIAL_CSV):
        return completed
    with open(FULL_TRIAL_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            completed.add((row["controller"], row["tier"], row["condition"], int(row["seed_idx"])))
    return completed


def append_row(row: dict) -> None:
    exists = os.path.exists(FULL_TRIAL_CSV)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(FULL_TRIAL_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main(n_seeds: int = N_SEEDS) -> None:
    assert n_seeds >= 30, "Blueprint v2 / IEEE requirement: >=30 held-out seeds per cell."
    controllers = build_controllers()
    controllers = add_fault_aware_condition(controllers)
    if not controllers:
        raise RuntimeError("No controllers available.")

    conditions = build_conditions()
    completed = load_completed()

    # Single source of truth (config.get_eval_trial_seeds) -- guarantees
    # every controller (A/B/D/E/fault-aware) is scored on IDENTICAL held-out
    # seeds, and that NSGA-II/ANN training code (which excludes these same
    # seeds via config.sample_disjoint_seeds) never trains on an evaluation
    # seed.
    trial_seeds = get_eval_trial_seeds(n_seeds=n_seeds)

    total = len(controllers) * len(TIERS) * len(conditions) * n_seeds
    done = 0

    for cname, controller in controllers.items():
        for tier in TIERS:
            for cond_name, fault_type, sev_range in conditions:
                for seed_idx, trial_seed in enumerate(trial_seeds):
                    key = (cname, tier, cond_name, seed_idx)
                    if key in completed:
                        done += 1
                        continue

                    if fault_type is None:
                        sim = RobotSimulator(tier, seed=trial_seed, faulty=False)
                    else:
                        spec = sample_fault_spec_typed(trial_seed, fault_type=fault_type,
                                                        severity_range=sev_range)
                        sim = RobotSimulator(tier, seed=trial_seed, faulty=True, spec_override=spec)

                    result = sim.run(controller)
                    mode = classify_failure_mode(result)
                    m = result.metrics

                    row = {
                        "controller": cname, "tier": tier, "condition": cond_name,
                        "fault_type": fault_type or "", "fault_severity": (
                            sev_range[1] if sev_range else ""),
                        "seed_idx": seed_idx, "trial_seed": trial_seed,
                        "success": m.success, "collision": m.collision,
                        "timeout": mode == "timeout",
                        "stuck_oscillation": mode == "stuck_oscillation",
                        "path_length": m.path_length, "time_to_goal": m.time_to_goal,
                        "min_clearance": m.min_clearance,
                        "steering_smoothness": m.steering_smoothness,
                        "final_goal_distance": m.final_goal_distance, "steps": m.steps,
                    }
                    append_row(row)
                    done += 1
                    if done % 25 == 0 or done == total:
                        print(f"[{done}/{total}] {cname} | {tier} | {cond_name} | "
                              f"seed_idx={seed_idx} -> {mode}")

    print(f"\nDone: {done}/{total} trials in {FULL_TRIAL_CSV}")


if __name__ == "__main__":
    main()
