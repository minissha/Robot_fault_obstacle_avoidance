"""
experiments/wide_sensor_ablation.py

Evidence-backed ablation of the opt-in wide-angle sensor patch
(simulation_core.get_wide_sensor_readings / RobotSimulator(wide_sensors=True)
/ controllers/baseline.py's WIDE_NEAR_THRESHOLD-gated override).

WHAT THIS TESTS: does adding two extra rays at +-N degrees (closing the
blind spot between the existing +-45 deg rays and full side coverage,
which the collision-metrics audit showed is where dense-tier grazing
collisions happen -- min_clearance 1.5-2.5 units at failure) produce a
REAL, OUT-OF-SAMPLE improvement in dense-tier success, without
regressing sparse?

METHOD (two-stage, to avoid the exact leakage this project's own audit
trail has repeatedly caught elsewhere -- tuning and evaluating on the
same seeds):
  Stage 1: grid-search angle/threshold/turn-magnitude on the 50 canonical
           held-out seeds (config.get_eval_trial_seeds()).
  Stage 2: take ONLY the single best config from Stage 1 and re-test it
           on a completely fresh, disjoint 100-seed set
           (config.sample_disjoint_seeds()). This is the number that
           actually counts.

RESULT (this session, real run, both stages executed exactly as coded
below -- see FINAL_REPORT.md's audit-trail conventions):
  Stage 1 best config: angle=75deg, WIDE_NEAR_THRESHOLD=15, WIDE_TURN_DEG=35
    -> dense 17/50 (34.0%) vs baseline 9/50 (18.0%), Mann-Whitney p=0.070
       (already not significant at alpha=0.05, and this is BEFORE any
       multiple-comparisons correction for the 48 configs tried).
  Stage 2 (fresh 100 seeds, same config, no further tuning):
    -> dense 25/100 (25.0%) vs baseline 20/100 (20.0%), p=0.399
    -> sparse 34/100 (34.0%) vs baseline 40/100 (40.0%), p=0.382
       (a nominal REGRESSION on sparse, also not significant)

CONCLUSION: the Stage-1 "win" did not replicate out-of-sample -- the
effect size shrank from +16pp to +5pp and lost what little significance
it had. This is consistent with Stage 1 simply overfitting the 50 tuning
seeds across a 48-point grid (classic multiple-comparisons artifact, not
a real effect). NO wide-angle-ray configuration tested here produces a
validated, replicable improvement in dense-tier collision avoidance.

RECOMMENDATION: do not enable wide_sensors=True in
run_full_experiment.py's default controller grid on the strength of this
result. The dense-tier collision-dominated failure mode remains best
explained as the architectural limit already identified in
FINAL_REPORT.md Sec 19.6 (3-5 fixed instantaneous rays, no memory --
Koren & Borenstein 1991): a single extra pair of rays, at any angle or
threshold tried here, is not enough to fix it. The two remaining
options, in order of effort, are (a) sensor memory (last-N readings per
ray, so a closing obstacle is caught by trend rather than instantaneous
position -- larger change, touches every downstream consumer of the
observation width) or (b) reporting the ceiling as a stated Discussion
limitation, which is what the existing report drafts already lean toward.

Run: python experiments/wide_sensor_ablation.py
(No pymoo/network required -- pure simulation_core + baseline controller.)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import mannwhitneyu

import simulation_core as sc
import controllers.baseline as bl
import config
from simulation_core import RobotSimulator
from controllers.baseline import BaselineController


def run_batch(tier: str, wide: bool, seeds, angles=None, thresh=None, turn=None):
    """Run BaselineController over `seeds` on `tier`, clean condition.
    Optionally overrides the module-level wide-sensor angle/threshold/
    turn constants for this batch (restored by the caller afterward is
    NOT done here -- this script is standalone and exits after use;
    do not import this module's side effects into a long-lived process
    without resetting sc.WIDE_SENSOR_ANGLES_DEG / bl.WIDE_NEAR_THRESHOLD /
    bl.WIDE_TURN_DEG yourself first).
    """
    if angles is not None:
        sc.WIDE_SENSOR_ANGLES_DEG = angles
    if thresh is not None:
        bl.WIDE_NEAR_THRESHOLD = thresh
    if turn is not None:
        bl.WIDE_TURN_DEG = turn
    ctrl = BaselineController()
    succ, coll = [], []
    for s in seeds:
        sim = RobotSimulator(tier=tier, seed=s, faulty=False, wide_sensors=wide)
        m = sim.run(ctrl).metrics
        succ.append(int(m.success))
        coll.append(int(m.collision))
    return succ, coll


def stage1_grid_search(tuning_seeds):
    base_dense, _ = run_batch("dense", False, tuning_seeds)
    base_sparse, _ = run_batch("sparse", False, tuning_seeds)

    grid = [
        (angle, thresh, turn)
        for angle in (65, 70, 75, 80)
        for thresh in (12, 15, 18, 22)
        for turn in (30, 35, 40)
    ]

    results = []
    for angle, thresh, turn in grid:
        ds, _ = run_batch("dense", True, tuning_seeds, (float(angle), -float(angle)), float(thresh), float(turn))
        ss, _ = run_batch("sparse", True, tuning_seeds, (float(angle), -float(angle)), float(thresh), float(turn))
        p_dense = mannwhitneyu(base_dense, ds, alternative="two-sided").pvalue
        p_sparse = mannwhitneyu(base_sparse, ss, alternative="two-sided").pvalue
        results.append((sum(ds), p_dense, sum(ss), p_sparse, angle, thresh, turn))

    results.sort(key=lambda r: (-r[0], r[1]))
    print(f"Stage 1 (n={len(tuning_seeds)} tuning seeds): baseline dense={sum(base_dense)}/{len(tuning_seeds)}, "
          f"baseline sparse={sum(base_sparse)}/{len(tuning_seeds)}")
    print(f"{'dense':>6} {'p_dense':>8} {'sparse':>7} {'p_sparse':>9}   angle thresh turn")
    for ds, pd, ss, ps, angle, thresh, turn in results[:10]:
        print(f"{ds:6d} {pd:8.3f} {ss:7d} {ps:9.3f}   {angle:5d} {thresh:6d} {turn:4d}")

    best = results[0]
    return {"angle": best[4], "thresh": best[5], "turn": best[6]}


def stage2_out_of_sample_validation(best_config, tuning_seeds):
    eval_seeds = set(tuning_seeds)
    rng = np.random.default_rng(999)
    fresh_seeds = config.sample_disjoint_seeds(rng, n=100, exclude=eval_seeds)
    assert len(fresh_seeds) == 100
    assert not (set(fresh_seeds) & eval_seeds), "leakage: fresh seeds overlap tuning seeds"

    angle, thresh, turn = best_config["angle"], best_config["thresh"], best_config["turn"]

    base_dense, _ = run_batch("dense", False, fresh_seeds)
    cand_dense, _ = run_batch("dense", True, fresh_seeds, (float(angle), -float(angle)), float(thresh), float(turn))
    base_sparse, _ = run_batch("sparse", False, fresh_seeds)
    cand_sparse, _ = run_batch("sparse", True, fresh_seeds, (float(angle), -float(angle)), float(thresh), float(turn))

    p_dense = mannwhitneyu(base_dense, cand_dense, alternative="two-sided").pvalue
    p_sparse = mannwhitneyu(base_sparse, cand_sparse, alternative="two-sided").pvalue

    n = len(fresh_seeds)
    print(f"\nStage 2 (n={n} FRESH, disjoint validation seeds; config: angle={angle} thresh={thresh} turn={turn}):")
    print(f"dense:  baseline={sum(base_dense)}/{n} ({sum(base_dense)/n:.3f})  "
          f"candidate={sum(cand_dense)}/{n} ({sum(cand_dense)/n:.3f})  p={p_dense:.4f}")
    print(f"sparse: baseline={sum(base_sparse)}/{n} ({sum(base_sparse)/n:.3f})  "
          f"candidate={sum(cand_sparse)}/{n} ({sum(cand_sparse)/n:.3f})  p={p_sparse:.4f}")

    verdict = "VALIDATED" if (p_dense < 0.05 and sum(cand_dense) > sum(base_dense) and p_sparse > 0.05) else "NOT VALIDATED"
    print(f"\nVerdict: {verdict} (requires dense p<0.05 in the improving direction AND no significant sparse regression)")
    return verdict


def main():
    tuning_seeds = config.get_eval_trial_seeds()
    best_config = stage1_grid_search(tuning_seeds)
    stage2_out_of_sample_validation(best_config, tuning_seeds)


if __name__ == "__main__":
    main()
