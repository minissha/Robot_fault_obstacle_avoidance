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
from typing import Optional

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
    "steering_smoothness", "final_goal_distance", "steps", "mean_speed",
]
CSV_FIELDS = ["controller", "tier", "condition", "fault_type", "fault_severity",
              "seed_idx", "trial_seed", "artifact_fingerprint"] + METRIC_FIELDS


def artifact_fingerprint() -> str:
    """
    A short hash of the trained files this run depends on (the detector,
    the NSGA-II result, the ANN), based on their size and last-modified
    time.

    The runner can resume, which is fine as long as the models haven't
    changed in between. If they have, the old rows were scored against a
    different detector and mixing them with new ones gives you a results
    file that quietly averages two different experiments. Stamping each row
    with this hash makes that detectable.
    """
    import hashlib
    parts = []
    for path in (FAULT_DETECTOR_PATH, NSGA2_PATH, ANN_MODEL_PATH):
        if os.path.exists(path):
            st = os.stat(path)
            parts.append(f"{os.path.basename(path)}:{st.st_mtime_ns}:{st.st_size}")
        else:
            parts.append(f"{os.path.basename(path)}:absent")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


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


def select_best_controller(candidate_names, fingerprint: str) -> Optional[str]:
    """
    Work out which controller actually did best on the clean runs, so the
    fault-aware comparison gets wrapped around the strongest one.

    This used to just assume the NSGA-II controller was best, which turned
    out not to be true, so the whole comparison was being built on one of
    the weaker controllers. Reads the rows this run already wrote, so it
    costs nothing extra and uses the same seeds everything else was scored
    on. Ties go alphabetically so it stays reproducible.

    Returns None if there's nothing to rank yet.
    """
    if not os.path.exists(FULL_TRIAL_CSV):
        return None

    wins: dict = {}
    counts: dict = {}
    with open(FULL_TRIAL_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("artifact_fingerprint") != fingerprint:
                continue
            if row["condition"] != "clean":
                continue
            name = row["controller"]
            if name not in candidate_names:
                continue
            wins[name] = wins.get(name, 0) + (1 if row["success"] == "True" else 0)
            counts[name] = counts.get(name, 0) + 1

    if not counts:
        return None

    rates = {n: wins[n] / counts[n] for n in counts}
    order = sorted(rates, key=lambda n: (-rates[n], n))   # rate desc, then name
    best = order[0]
    ranking = ", ".join(f"{n}={rates[n]:.3f}" for n in order)
    print(f"Clean-condition ranking (pooled over tiers): {ranking}")
    print(f"Best fault-blind controller by measured clean success: '{best}'.")
    return best


def add_fault_aware_condition(controllers: dict, best_name: str) -> dict:
    """Sec 6.2's 5th condition: wrap the MEASURED best-performing controller
    (see select_best_controller) as fault-aware, so it can be compared 1:1
    against its own fault-blind row under identical seeds and faults."""
    if not os.path.exists(FAULT_DETECTOR_PATH):
        print(f"[skip] {FAULT_DETECTOR_PATH} not found -- run "
              f"experiments/train_fault_detector.py first.")
        return {}
    if best_name not in controllers:
        print(f"[skip] best controller '{best_name}' not in the controller set.")
        return {}

    detector = FaultDetector.load(FAULT_DETECTOR_PATH)
    base = controllers[best_name]
    print(f"Added fault-aware condition wrapping '{best_name}'.")
    return {best_name + FAULT_AWARE_SUFFIX: FaultAwareController(base, detector)}


def load_completed(current_fingerprint: str) -> set:
    """Only counts a row as completed if it was written under the CURRENT
    artifact fingerprint (see artifact_fingerprint() docstring)."""
    completed = set()
    if not os.path.exists(FULL_TRIAL_CSV):
        return completed
    with open(FULL_TRIAL_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("artifact_fingerprint") != current_fingerprint:
                continue
            completed.add((row["controller"], row["tier"], row["condition"], int(row["seed_idx"])))
    return completed


def purge_stale_rows(current_fingerprint: str) -> int:
    """
    Clear out rows from earlier runs before writing any new ones.

    The resume check already ignored old rows when deciding what still
    needed running, but it left them sitting in the file, so every re-run
    added another full copy of the results. At one point the file had 9,800
    rows for 4,900 actual trials. The figures ended up averaging both copies
    while the stats script read only one, which is how the same number
    showed up three different ways.

    Old rows can't be salvaged anyway since they were scored against
    different models, so dropping them loses nothing.

    Returns how many rows were dropped.
    """
    if not os.path.exists(FULL_TRIAL_CSV):
        return 0

    with open(FULL_TRIAL_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_FIELDS:
            return 0          # schema mismatch is handled by append_row
        rows = list(reader)

    keep = [r for r in rows if r.get("artifact_fingerprint") == current_fingerprint]
    dropped = len(rows) - len(keep)
    if dropped == 0:
        return 0

    stale = {}
    for r in rows:
        fp = r.get("artifact_fingerprint")
        if fp != current_fingerprint:
            stale[fp] = stale.get(fp, 0) + 1
    detail = ", ".join(f"{fp}={n} rows" for fp, n in sorted(stale.items()))
    print(f"Purging {dropped} stale row(s) from {FULL_TRIAL_CSV} ({detail}); "
          f"keeping {len(keep)} row(s) at fingerprint {current_fingerprint}.")

    with open(FULL_TRIAL_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(keep)
    return dropped


def append_row(row: dict) -> None:
    exists = os.path.exists(FULL_TRIAL_CSV)
    if exists:
        with open(FULL_TRIAL_CSV, "r", newline="") as f:
            existing_header = f.readline().strip().split(",")
        if existing_header != CSV_FIELDS:
            raise RuntimeError(
                f"{FULL_TRIAL_CSV} has an outdated column schema (missing "
                f"'artifact_fingerprint' or otherwise stale). Delete it and "
                f"re-run for a clean, unambiguous result set -- silently "
                f"appending mismatched columns would corrupt the CSV."
            )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(FULL_TRIAL_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_grid(controllers: dict, conditions: list, trial_seeds: list,
             fingerprint: str, completed: set, label: str) -> int:
    """Run every (controller, tier, condition, seed) cell not already
    completed, appending one row each. Factored out of main() so the
    fault-blind phase and the fault-aware phase share identical logic."""
    total = len(controllers) * len(TIERS) * len(conditions) * len(trial_seeds)
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
                        "artifact_fingerprint": fingerprint,
                        "success": m.success, "collision": m.collision,
                        "timeout": mode == "timeout",
                        "stuck_oscillation": mode == "stuck_oscillation",
                        "path_length": m.path_length, "time_to_goal": m.time_to_goal,
                        "min_clearance": m.min_clearance,
                        "steering_smoothness": m.steering_smoothness,
                        "final_goal_distance": m.final_goal_distance, "steps": m.steps,
                        "mean_speed": m.mean_speed,
                    }
                    append_row(row)
                    done += 1
                    if done % 25 == 0 or done == total:
                        print(f"[{label} {done}/{total}] {cname} | {tier} | {cond_name} | "
                              f"seed_idx={seed_idx} -> {mode}")
    return done


def main(n_seeds: int = N_SEEDS) -> None:
    assert n_seeds >= 30, "Blueprint v2 / IEEE requirement: >=30 held-out seeds per cell."
    controllers = build_controllers()
    if not controllers:
        raise RuntimeError("No controllers available.")

    conditions = build_conditions()
    fingerprint = artifact_fingerprint()
    print(f"Artifact fingerprint for this run: {fingerprint}")
    purge_stale_rows(fingerprint)
    completed = load_completed(fingerprint)

    # Single source of truth (config.get_eval_trial_seeds) -- guarantees
    # every controller (A/B/D/E/fault-aware) is scored on IDENTICAL held-out
    # seeds, and that NSGA-II/ANN training code (which excludes these same
    # seeds via config.sample_disjoint_seeds) never trains on an evaluation
    # seed.
    trial_seeds = get_eval_trial_seeds(n_seeds=n_seeds)

    # Phase 1: every fault-blind controller.
    done = run_grid(controllers, conditions, trial_seeds, fingerprint,
                    completed, label="fault-blind")

    # Phase 2: wrap the controller that MEASURABLY performed best in the
    # clean condition (not a hardcoded guess) and run it as the fault-aware
    # 5th condition. Phase 1's rows are already on disk, so this ranking
    # costs no extra simulation and uses the same held-out seeds.
    best_name = select_best_controller(set(controllers), fingerprint)
    if best_name is not None:
        fault_aware = add_fault_aware_condition(controllers, best_name)
        if fault_aware:
            completed = load_completed(fingerprint)
            done += run_grid(fault_aware, conditions, trial_seeds, fingerprint,
                             completed, label="fault-aware")

    print(f"\nDone: {done} trials in {FULL_TRIAL_CSV}")


if __name__ == "__main__":
    main()
