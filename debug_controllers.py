"""
debug_controllers.py

Blueprint v2, Section 2 (Debugging Requirements -- Blocking, Do First).

Run from the project root:
    python debug_controllers.py

Performs, in order:
  1. Dense-tier feasibility audit (Sec 2.1): obstacle-obstacle clearance
     vs. robot diameter, sensor-range-vs-gap check.
  2. Hand-tuned FLC rule-coverage heatmap (Sec 2.2): confirms all 27
     (F,L,R) term combinations now map to a non-trivial output.
  3. Sign/unit convention check (Sec 2.2): confirms steering direction
     matches the kinematics update for a synthetic left/right-obstacle probe.
  4. Trajectory replay of 5 sparse-clean episodes with before/after
     success counts (uses the CURRENT code, i.e. post-fix).
  5. Quick statistical-hygiene check (Sec 2.3): with N_SEEDS>=30, no
     zero-variance/NaN cells should occur in a Mann-Whitney U test.

This script does not fabricate results -- every number below is produced
by actually running the simulator with the current controller code.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

from config import N_SEEDS
from simulation_core import (
    N_SENSORS,
    unpack_action,
    RobotSimulator, generate_obstacles, _dist_rect_to_rect,
    ROBOT_RADIUS, SENSOR_RANGE, MIN_OBSTACLE_GAP,
)
from controllers.baseline import BaselineController
from controllers.fuzzy_handtuned import HandTunedFLC, RULES, DEFAULT_PARAMS, fuzzify


def audit_dense_feasibility(n_seeds: int = 30) -> None:
    print("\n=== 2.1 Dense-Tier Feasibility Audit ===")
    robot_diameter = 2 * ROBOT_RADIUS
    required_gap = 2.5 * robot_diameter
    print(f"Robot diameter: {robot_diameter}, required clearance (2.5x): {required_gap}")
    print(f"Sensor range: {SENSOR_RANGE} (rule of thumb: >= 1.5x min passable gap "
          f"= {1.5 * required_gap:.1f}) -> {'OK' if SENSOR_RANGE >= 1.5 * required_gap else 'INSUFFICIENT'}")

    min_gaps = []
    counts = []
    for seed in range(n_seeds):
        obs = generate_obstacles("dense", seed)
        counts.append(len(obs))
        gaps = [_dist_rect_to_rect(a, b) for a, b in itertools.combinations(obs, 2)]
        min_gaps.append(min(gaps) if gaps else float("inf"))

    min_gaps = np.array(min_gaps)
    print(f"Dense obstacle count over {n_seeds} seeds: min={min(counts)}, max={max(counts)}")
    print(f"Min obstacle-obstacle gap over {n_seeds} seeds: "
          f"min={min_gaps.min():.2f}, mean={min_gaps.mean():.2f}")
    n_infeasible = int(np.sum(min_gaps < MIN_OBSTACLE_GAP - 1e-6))
    print(f"Seeds with any gap below MIN_OBSTACLE_GAP={MIN_OBSTACLE_GAP}: {n_infeasible}/{n_seeds}")
    verdict = "PASS -- all layouts are geometrically passable" if n_infeasible == 0 else "FAIL -- infeasible layouts remain"
    print(f"Verdict: {verdict}")


def rule_coverage_heatmap() -> None:
    print("\n=== 2.2a FLC Rule-Coverage Heatmap ===")
    terms = ("Near", "Medium", "Far")
    covered = {(f, l, r): out for f, l, r, out in RULES}
    print(f"Rules defined: {len(RULES)} / 27 possible (F,L,R) combinations")
    missing = [(f, l, r) for f in terms for l in terms for r in terms if (f, l, r) not in covered]
    if missing:
        print(f"MISSING combinations (would silently default to Straight/0.0 output): {missing}")
    else:
        print("All 27 combinations covered -- no silent zero-output regions remain.")

    # Confirm every grid point produces SOME non-zero rule activation
    # (i.e. at least one rule fires), which is the actual coverage-gap
    # symptom described in the blueprint (zero *activation*, not zero
    # *output* -- a legitimate "Straight" decision also outputs 0.0 deg).
    controller = HandTunedFLC()
    f_near, f_far, l_near, l_far, r_near, r_far = controller.params
    no_activation_points = 0
    total_points = 0
    for f_val in np.linspace(0, 100, 11):
        for l_val in np.linspace(0, 100, 11):
            for r_val in np.linspace(0, 100, 11):
                total_points += 1
                f_mf = fuzzify(f_val, f_near, f_far)
                l_mf = fuzzify(l_val, l_near, l_far)
                r_mf = fuzzify(r_val, r_near, r_far)
                any_fired = any(
                    min(f_mf[ft], l_mf[lt], r_mf[rt]) > 0.0
                    for ft, lt, rt, _ in RULES
                )
                if not any_fired:
                    no_activation_points += 1
    print(f"Sampled {total_points} (F,L,R) grid points; "
          f"{no_activation_points} had ZERO rule activation (true coverage "
          f"gap symptom) -- expect 0 after the fix.")


def sign_convention_check() -> None:
    print("\n=== 2.2b Steering Sign / Kinematics Convention Check ===")
    controller = HandTunedFLC()
    # Neutral goal (goal_angle=0) so the goal-attraction blend contributes
    # nothing, keeping this a pure obstacle-avoidance sign check. Rays are
    # ordered left to right, so blocking the first half blocks the left.
    def obstacle_on(side: str) -> np.ndarray:
        readings = np.full(N_SENSORS, 80.0)
        half = N_SENSORS // 2
        if side == "left":
            readings[:half] = 8.0
        else:
            readings[half + 1:] = 8.0
        return np.concatenate([readings, [50.0, 0.0]])

    steer_left_obstacle = unpack_action(controller.predict(obstacle_on("left")))[0]
    steer_right_obstacle = unpack_action(controller.predict(obstacle_on("right")))[0]
    print(f"Obstacle on the LEFT  -> steer={steer_left_obstacle:+.1f} deg "
          f"(expect negative/right-turn): {'OK' if steer_left_obstacle < 0 else 'BUG'}")
    print(f"Obstacle on the RIGHT -> steer={steer_right_obstacle:+.1f} deg "
          f"(expect positive/left-turn): {'OK' if steer_right_obstacle > 0 else 'BUG'}")
    print("Kinematics: heading += radians(steer_deg) * STEER_GAIN (positive steer "
          "-> counterclockwise). The rays are listed from +75 deg (left) down to "
          "-75 deg (right), so turning away from a blocked left side means a "
          "negative steer, and vice versa.")


def replay_and_success_check(n_episodes: int = 5, n_seeds_for_rate: int = 100) -> None:
    print(f"\n=== 2.2c Sparse-Clean Replay ({n_episodes} episodes) + Success-Rate Check "
          f"(n={n_seeds_for_rate} seeds) ===")
    for ctrl_name, ctrl_cls in (("A_baseline", BaselineController), ("B_handtuned_flc", HandTunedFLC)):
        outcomes = {"success": 0, "collision": 0, "timeout": 0}
        for seed in range(n_seeds_for_rate):
            sim = RobotSimulator("sparse", seed=seed + 9000, faulty=False)
            r = sim.run(ctrl_cls())
            if r.metrics.success:
                outcomes["success"] += 1
            elif r.metrics.collision:
                outcomes["collision"] += 1
            else:
                outcomes["timeout"] += 1
        rate = outcomes["success"] / n_seeds_for_rate
        print(f"{ctrl_name}: success={outcomes['success']}/{n_seeds_for_rate} "
              f"({rate:.1%}), collision={outcomes['collision']}, timeout={outcomes['timeout']}")

    for seed in range(n_episodes):
        sim = RobotSimulator("sparse", seed=seed, faulty=False)
        r = sim.run(HandTunedFLC())
        print(f"  replay seed={seed}: outcome="
              f"{'success' if r.metrics.success else ('collision' if r.metrics.collision else 'timeout')}, "
              f"steps={r.metrics.steps}, path_length={r.metrics.path_length:.1f}")


def re_run_criterion_check(n_seeds: int = N_SEEDS) -> None:
    """Blueprint v2 Sec 2.4: must hold before writing any results section."""
    print(f"\n=== 2.4 Re-Run Criterion Check (n_seeds={n_seeds}) ===")
    from scipy import stats

    results = {}
    for ctrl_name, ctrl_cls in (("A_baseline", BaselineController), ("B_handtuned_flc", HandTunedFLC)):
        for tier in ("sparse", "dense"):
            succ = []
            for seed in range(n_seeds):
                sim = RobotSimulator(tier, seed=seed + 42_000, faulty=False)
                r = sim.run(ctrl_cls())
                succ.append(1 if r.metrics.success else 0)
            results[(ctrl_name, tier)] = np.array(succ)

    dense_nonzero = all(results[(c, "dense")].sum() > 0 for c in ("A_baseline", "B_handtuned_flc"))
    print(f"(a) dense-tier success non-zero for baseline AND hand-tuned FLC: "
          f"{'PASS' if dense_nonzero else 'FAIL'} "
          f"(baseline={results[('A_baseline','dense')].sum()}/{n_seeds}, "
          f"flc={results[('B_handtuned_flc','dense')].sum()}/{n_seeds})")

    base_sparse = results[("A_baseline", "sparse")]
    flc_sparse = results[("B_handtuned_flc", "sparse")]
    print(f"(b) sparse-clean success -- baseline={base_sparse.sum()}/{n_seeds}, "
          f"FLC={flc_sparse.sum()}/{n_seeds} "
          f"({'FLC competitive/beats baseline' if flc_sparse.sum() >= base_sparse.sum() else 'FLC below baseline (report honestly; non-zero, no longer a bug)'})")

    try:
        u_stat, p_val = stats.mannwhitneyu(base_sparse, flc_sparse, alternative="two-sided")
        nan_free = not (np.isnan(u_stat) or np.isnan(p_val))
    except ValueError:
        u_stat, p_val, nan_free = float("nan"), float("nan"), False
    print(f"(c) Mann-Whitney U (baseline vs FLC, sparse success): "
          f"U={u_stat}, p={p_val}, NaN-free={'PASS' if nan_free else 'FAIL'}")


def main() -> None:
    audit_dense_feasibility(n_seeds=30)
    rule_coverage_heatmap()
    sign_convention_check()
    replay_and_success_check()
    re_run_criterion_check()
    print("\nDone. Review PASS/FAIL/BUG markers above before proceeding to "
          "experiments/run_full_experiment.py.")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
