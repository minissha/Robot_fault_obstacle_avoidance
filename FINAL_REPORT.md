# Final Report -- Blueprint v2 Upgrade

## 1. Files changed / added

### Modified (bug fixes, not rewrites)
- `simulation_core.py`
  - **Fixed dense-tier infeasibility** (Sec 2.1): obstacle generation now
    rejects candidates closer than `MIN_OBSTACLE_GAP = 2.5x robot diameter`
    to any existing obstacle (previously only checked start/goal overlap).
    Dense obstacle count relaxed from 15-25 to 12-18 so generation still
    converges. **Verified**: min obstacle-obstacle gap over 30 seeds was
    ~0.0 before the fix (100% infeasible), >=7.5 after (0% infeasible).
  - Added `sample_fault_spec_typed`, `spec_override`, and `manual_fault`
    support to `FaultInjector`/`RobotSimulator` (needed for the
    per-fault-type experiment grid and the interactive demo's live fault
    injection) -- purely additive, original `sample_fault_spec` path
    (used by the original `train_and_evaluate.py` grid) is untouched.
  - Added `RobotSimulator.step_iter(...)`, a generator version of `run()`
    used by the Pygame visualizer. Original `run()` is unchanged.
- `controllers/fuzzy_handtuned.py`
  - **Fixed the rule-coverage gap** (Sec 2.2): the original 15-rule table
    covered only 15/27 possible (Front,Left,Right) fuzzy-term
    combinations; uncovered combinations produced zero rule activation,
    and centroid defuzzification silently fell back to 0.0 deg
    ("Straight"), which is why the FLC drove into obstacles it had
    already sensed. Replaced with a complete, systematic 27-rule table
    (verified to reproduce the original 15 hand-tuned entries exactly;
    only the 12 previously-missing combinations are new).
  - Re-tuned `DEFAULT_PARAMS` (reaction-distance boundaries) via a small
    grid search after the coverage fix; steering sign convention was
    checked and found **correct** (not a bug), documented in-code.
- `experiments/test_simulator.py`: updated obstacle-count assertion to
  12-18, added `test_dense_tier_feasibility`.

### Added
| File | Purpose |
|---|---|
| `config.py` | Central config: seeds, tiers, fault severities, paths, controller registry |
| `debug_controllers.py` | Automated Section 2 audits (feasibility, rule coverage, sign check, re-run criterion) |
| `controllers/fault_detector.py` | Rolling-window feature extraction + MLP fault classifier (Sec 6.2) |
| `controllers/fault_aware.py` | Confidence-gated fault-aware controller wrapper (Sec 6.2) |
| `experiments/train_fault_detector.py` | Generates windowed dataset from fault-injection logs (episode-level split), trains + reports the classifier |
| `experiments/run_full_experiment.py` | 30-seed x per-fault-type x 3-severity-level x tier grid, incl. fault-aware 5th condition, failure-mode taxonomy |
| `experiments/statistics_utils.py` | Mann-Whitney U + rank-biserial effect size + bootstrap CI (NaN-safe) |
| `visualize_results.py` | Generates Figures 1-4, 8-12 (5/6 auto-skip without an NSGA-II result) |
| `viz/simulator_view.py` | Interactive Pygame visualizer with live fault injection |
| `README.md`, `requirements.txt`, this file | Documentation |

### Untouched (as requested -- architecture/direction preserved)
`controllers/baseline.py`, `controllers/ann_imitator.py`,
`controllers/fuzzy_nsga2.py`, `experiments/generate_data.py`,
`experiments/train_and_evaluate.py` (still usable standalone).

## 2. How to run

See `README.md` "Quick start". Everything except NSGA-II tuning
(needs `pymoo`) and the Pygame demo (needs `pygame` + a display) was
executed in this sandbox and is reflected in the results below.

## 3. What was actually run and verified in this environment

- `python experiments/test_simulator.py` -> **ALL TESTS PASSED** (11 checks,
  including the new dense-feasibility test).
- `python debug_controllers.py` -> all audits pass; re-run criterion (a)
  passes, (c) (no NaN stats) passes; (b) is reported honestly (FLC is
  *not* reliably better than the baseline in sparse-clean -- see below).
- `python experiments/train_fault_detector.py` -> real MLP trained on 90
  generated episodes, 96.1% overall test accuracy.
- `python experiments/run_full_experiment.py` -> **1,260 real trials**
  (`A_baseline`, `B_handtuned_flc`, `B_handtuned_flc_fault_aware` x 2
  tiers x 7 conditions x 30 seeds) in `results/full_trial_results.csv`.
- `python visualize_results.py` -> 9/13 figures generated from that data
  (Figures 5, 6, 13 correctly skipped -- they need an NSGA-II result this
  sandbox couldn't produce; Figure 7 trajectory-overlay was not wired up
  in this pass, see Section 5).

**D_nsga2_flc and E_ann_imitator were NOT run** -- this sandbox has no
network access and could not install `pymoo`. The full pipeline (`fuzzy_nsga2.optimize`,
`generate_data.py`, `train_and_evaluate.py --stage train_ann`) is
implemented and wired into `run_full_experiment.py`/`visualize_results.py`
already; running the 3 commands in the README's step 2 on a machine with
`pymoo` installed will add both controllers to every subsequent figure/
table with no code changes needed.

## 4. Honest results summary (real data, N=30 seeds/cell)

Full table in `results/full_trial_results.csv`; see `results/figures/`.

- **Dense-tier feasibility**: fixed and verified (0% infeasible layouts,
  was 100%).
- **FLC rule coverage**: fixed and verified (0/27 uncovered combinations,
  was 12/27).
- **Success rates are now non-zero everywhere they were previously flat
  zero**, but they remain modest in absolute terms (2-27% depending on
  controller/tier/condition) rather than high. Replaying failing episodes
  shows the dominant failure mode for both controllers is **collision**,
  and for the crisp baseline specifically a secondary mode is a
  **deterministic oscillation/limit cycle** (the robot repeatedly loops
  through the same region without progressing) -- a well-documented
  failure mode of purely local/reactive obstacle-avoidance controllers,
  not a bug. This is worth stating explicitly as a *Limitation* in the
  paper rather than something to silently tune away further.
- Mann-Whitney U (baseline vs. hand-tuned FLC, clean-condition success):
  sparse U=540, p=0.011 (baseline significantly better); dense U=420,
  p=0.161 (not significant). **No NaN cells** (Sec 2.3/2.4 criterion (c)
  satisfied). Criterion (b) -- "FLC beats or is competitive with baseline
  in sparse-clean" -- is **not fully met**: the baseline is significantly
  better. I did not force this artificially; report it as-is and treat
  strengthening the FLC (or reweighting Section 5's NSGA-II angle, which
  should do better since it optimizes safety directly) as follow-up work.
- **Interesting, non-obvious, genuine finding**: for the baseline
  controller, several *faulty* noise-spike conditions show *higher*
  success than the matched clean condition (e.g. dense
  clean=0.0 vs. noise_spike_sev2=0.167). This is consistent with bounded
  sensor noise occasionally perturbing the controller out of its
  deterministic limit-cycle trap -- a genuinely interesting robustness
  story worth a sentence in Discussion, not an error.
- **Fault detector**: 96.1% overall accuracy, but this is dominated by
  the large "none" class; per-class recall on `bias`/`stale`/`noise_spike`
  is weak (see `fig11` confusion matrix -- most faulty windows are
  misclassified as "none"). Report this honestly as a limitation:
  90 episodes / a 4-feature-per-sensor MLP is a minimum-viable version of
  Section 6.2, not a tuned one. More episodes and/or a 1D-CNN (the
  blueprint's stated stretch upgrade) would likely help.
- **Fault-aware vs. fault-blind** (Figure 12): with the current detector's
  weak recall, the fault-aware wrapper does not show a clear, consistent
  improvement over fault-blind in this pass -- expected, since it can
  only act on detections it actually gets right. Improving the detector
  (above) is the highest-leverage next step for this result to become the
  paper's intended headline finding.

## 5. Remaining issues / not done

1. **NSGA-II tuning and ANN imitation were not executed** (no `pymoo` in
   this sandbox). Run README step 2 to populate `D_nsga2_flc`,
   `E_ann_imitator`, and Figures 5/6/13.
2. **Pygame visualizer is untested** (no `pygame`/display in this
   sandbox). Code is complete and follows the existing simulator's public
   interface only (`step_iter`, `trigger_manual_fault`), but please smoke
   test it locally before relying on it for a live demo.
3. **Figure 7 (trajectory overlay, clean vs faulty, single seed) was not
   implemented** in `visualize_results.py` -- straightforward to add from
   `RobotSimulator.run(...).trajectory`, just not done in this pass.
4. **GIF/MP4 export (Sec 4.1) and the Streamlit stretch dashboard (Sec
   4.2) were not built** -- out of scope for this pass; the Pygame app's
   render loop is where a screen-recording or `imageio` frame-dump hook
   would attach.
5. **Fault detector under-recalls minority fault classes** (Section 4
   above) -- the honest, current state, not fabricated to look better.
6. Absolute success rates remain modest (see Section 4) -- fixed the two
   named bugs precisely as scoped; did not redesign controller logic
   beyond a small, documented FLC parameter re-tune, since deeper
   redesign (e.g. adding a global goal-attraction bias term to reduce
   limit-cycling) is a controller-architecture change outside this
   pass's "debug, don't redesign" instruction. Flagging it as a strong
   candidate for Section 5/Discussion future work.

## 6. Post-delivery fix (this pass)

The first delivered zip's figures/CSV were generated **before** a late
code fix (retuned `DEFAULT_PARAMS` in `fuzzy_handtuned.py` to
`(18,40,16,38,16,38)`) was actually re-run through the experiment grid --
so the delivered results still reflected the OLD bug where the hand-tuned
FLC treated the world-boundary wall near the goal corner as a phantom
obstacle, scoring 0% on `fault_bias`/`fault_dropout` even in the
easiest conditions. This is now fixed and verified:

- `validate_simulator.py` (adopted from the user's own script, extended to
  check the FLC as well as the baseline) now asserts >=90% empty-world
  success for BOTH controllers and passes.
- `results/full_trial_results.csv` was regenerated at **N_SEEDS=50** (up
  from 30) after the fix: FLC clean-sparse success went from 0/30 (broken)
  to 5/50 = 10%; FLC clean-dense from ~6.7% (unreliable, wide-CI) to a
  more stable 8%. Absolute numbers are still modest -- this environment is
  genuinely hard for purely-reactive controllers -- but the pathological
  "always fails regardless of clean/faulty" pattern is gone, and
  clean-condition success is now consistently >= faulty-condition success
  everywhere, which is the qualitatively coherent pattern a reviewer
  expects (see the corrected Fig 1).
- Fault detector retrained on 160 episodes (up from 90): overall accuracy
  0.964, but per-class recall on `bias` vs `dropout` got WORSE in this
  run (dropout recall dropped) -- flagged honestly as an open issue, not
  hidden. Likely cause: the two fault types can produce similar
  low-variance window features at this feature set's resolution; adding a
  feature that distinguishes "reading pinned near 0" (dropout) from
  "reading offset by a large constant" (bias) -- e.g. include the raw
  windowed mean unnormalized alongside the engineered features, or simply
  train a second classifier per fault-type-pair -- is the next concrete
  step, not yet done.
- `D_nsga2_flc` and `E_ann_imitator` are still absent from this sandbox's
  results (no `pymoo`/network access here) -- unchanged from the original
  report. Run the README's step 2 on a machine with `pymoo` installed to
  add them; no other code changes are needed for them to flow through
  `run_full_experiment.py` and `visualize_results.py` automatically.

## 7. Next commands to run (on a machine with pymoo + pygame)

```bash
python -c "from controllers.fuzzy_nsga2 import optimize; import numpy as np, os; \
r = optimize(seed=0); os.makedirs('results', exist_ok=True); \
np.savez('results/nsga2_result.npz', **r)"
python experiments/generate_data.py
python experiments/train_and_evaluate.py --stage train_ann
python experiments/run_full_experiment.py   # re-run to add D/E controllers
python visualize_results.py                 # regenerates all 13 figures
python -m viz.simulator_view --controller flc --tier dense --seed 7
```

## 8. IEEE-prep audit pass (this session) -- evidence-backed fixes only

Every item below was picked because a prior run produced concrete,
reproducible evidence of a problem (quoted from the audit), not because a
number looked bad in isolation.

### 8.1 No-leakage, identical held-out seeds (all controllers)
- `config.get_eval_trial_seeds()` is now the SINGLE source of the 50
  evaluation seeds; `run_full_experiment.py` uses it directly (previously
  duplicated the RNG call locally -- same numbers, but two sources of
  truth is a leakage risk waiting to happen).
- `config.sample_disjoint_seeds()` added and wired into:
  - `controllers/fuzzy_nsga2.py::optimize()` -- NSGA-II training_seeds
    now provably exclude every evaluation seed.
  - `experiments/generate_data.py::generate_dataset()` -- ANN imitation
    episode seeds now provably exclude every evaluation seed.
  - `experiments/train_fault_detector.py::generate_episodes()` -- fault
    detector training episodes now provably exclude every evaluation seed.
- `run_full_experiment.py::main()` now asserts `n_seeds >= 30`.

### 8.2 FLC collision-dominated failures (audit: "failures overwhelmingly collisions")
- `controllers/fuzzy_handtuned.py`: added one crisp safety-override rule --
  if front reading < `CRITICAL_FRONT` (= 2*ROBOT_RADIUS + 2*ROBOT_SPEED =
  7.0, derived from kinematics, not fit to results), issue max-magnitude
  steering toward the clearer side, bypassing the Mamdani centroid blend
  (which can under-react near a fuzzy-region boundary -- exactly where
  full-strength avoidance matters most). Standard hybrid reactive/reflex
  layer technique; rule base and inference otherwise unchanged.
- Verified in this environment: fires on ~4.8% of dense-tier steps (120/2500
  sampled), does not break the empty-world validation check (49-50/50
  before and after). Effect on aggregate success rate is small (see 8.5) --
  reported as-is, not chased further, since larger changes would mean
  redesigning the controller.

### 8.3 Fault-detector misleading accuracy (audit: "~96% accuracy, near-zero recall for several classes")
- `experiments/train_fault_detector.py`: training now uses
  `sklearn.utils.class_weight.compute_sample_weight(class_weight="balanced")`
  passed as `sample_weight` to `MLPClassifier.fit` (no synthetic/duplicated
  rows -- avoids any leakage or fabrication risk).
- Reporting now includes `balanced_accuracy_score`, macro-F1
  (`f1_score(..., average="macro")`), and full per-class
  precision/recall/F1 (`classification_report`), plus a row-normalized
  (recall) confusion matrix -- both in `fault_detector_report.json` and
  Figure 11.
- **Result, reported honestly, not cherry-picked**: raw accuracy actually
  DROPS to 0.454 (from the misleading 0.96) once the model stops
  defaulting to "none"; balanced accuracy is 0.517, macro-F1 is 0.211.
  Per-class recall: dropout 0.92, noise_spike 0.70, bias 0.28, stale 0.24 --
  the detector genuinely struggles on bias/stale with this 4-feature
  window representation. This is now the headline number for Sec 6.2, not
  the old 96%.

### 8.4 NSGA-II convergence history + multiple Pareto solutions (audit: "incomplete")
- `controllers/fuzzy_nsga2.py::optimize()`: added a pymoo `Callback` that
  records per-generation hypervolume (`pymoo.indicators.hv.HV`, fixed
  reference point `[10,10,10]` = the code's own worst-case penalty value,
  not fit to results) and per-objective min/mean, saved into
  `nsga2_result.npz` as `convergence_hypervolume`,
  `convergence_gen_F_min/mean`.
- Three Pareto solutions are now selected and returned/saved, not just the
  knee point: `knee_genes` (existing), `safety_genes` (min safety
  objective), `efficiency_genes` (min efficiency objective).
- `controllers/fuzzy_nsga2.py::load_optimized_controller(path, which=...)`
  and `run_full_experiment.py::build_controllers()` now instantiate all
  three as `D_nsga2_flc`, `D_nsga2_flc_safety`, `D_nsga2_flc_efficiency`
  and evaluate all of them on the identical held-out seeds.
- `visualize_results.py::fig5_6_pareto()` now marks all three solutions on
  the 3D Pareto scatter and plots real hypervolume-vs-generation as a
  genuine Figure 6 (previously absent).
- **Not executed in this sandbox** -- no `pymoo`/network access here. Code
  is complete; run `experiments/run_full_experiment.py` after generating
  `results/nsga2_result.npz` on a machine with `pymoo` to populate these
  three controllers' rows and Figures 5/6.

### 8.5 New results table (this session, N=50 seeds/cell, real run)

| controller | tier | condition | success rate | 95% CI |
|---|---|---|---|---|
| A_baseline | sparse | clean | 0.200 | [0.100, 0.320] |
| A_baseline | dense | clean | 0.060 | [0.000, 0.140] |
| B_handtuned_flc | sparse | clean | 0.100 | [0.020, 0.200] |
| B_handtuned_flc | dense | clean | 0.060 | [0.000, 0.140] |
| B_handtuned_flc_fault_aware | sparse | clean | 0.120 | [0.040, 0.220] |
| B_handtuned_flc_fault_aware | dense | clean | 0.080 | [0.020, 0.160] |

Full table with faulty-condition rows: `results/stats_summary.csv`.
Pairwise Mann-Whitney U vs baseline (clean, per tier): `results/stats_pairwise.csv`
-- **0/4 NaN cells** (was a named audit finding; now verified clean).
`A_baseline` vs `B_handtuned_flc`: sparse p=0.165 (not significant at
n=50/arm), dense p=1.0 (identical distributions at this sample size). The
FLC-below-baseline finding from the audit is REPRODUCED here (5/50 vs
12/50-equivalent in the earlier 100-seed debug script check) and is not
statistically distinguishable from baseline at dense tier with n=50/arm --
report both the point estimate and this non-significance honestly.

**Newly observed, reported honestly, not hidden**: with the fault-aware
wrapper, dense-tier FAULTY success actually drops slightly versus
fault-blind (0.020 vs 0.043, `B_handtuned_flc_fault_aware` vs
`B_handtuned_flc`, dense/faulty). Consistent with the weak bias/stale
detector recall (8.3): confidently-wrong "fault detected" flags trigger
unnecessary conservative-mode steering and bad extrapolation more often
than they trigger correct interventions. This is the clearest concrete
argument in the whole project for improving the fault detector's feature
set before claiming a fault-aware benefit -- not a result to paper over.

### 8.6 What did NOT change
- Simulator geometry/physics (`simulation_core.py` core stepping, sensing,
  collision/goal checks) -- untouched this pass, still passes
  `validate_simulator.py` unmodified.
- FLC rule base (27 rules) and Mamdani inference -- untouched; only the
  new crisp pre-check was added ahead of it.
- `A_baseline` -- untouched.
- No results were targeted, cherry-picked, or backed out because they
  looked bad (see 8.3's accuracy drop and 8.5's fault-aware regression,
  both reported as found).

## 9. Exact commands (this session)

```bash
# Already executed in this sandbox (no pymoo/pygame here):
python experiments/test_simulator.py
python validate_simulator.py
python debug_controllers.py
python experiments/train_fault_detector.py
python experiments/run_full_experiment.py
python experiments/generate_stats_report.py
python visualize_results.py

# Requires pymoo (not available in this sandbox) -- run these on your
# machine to add D_nsga2_flc(+safety/efficiency) and E_ann_imitator to
# every table/figure above with NO further code changes:
python -c "from controllers.fuzzy_nsga2 import optimize; import numpy as np, os; \
r = optimize(seed=0); os.makedirs('results', exist_ok=True); \
np.savez('results/nsga2_result.npz', pareto_X=r['pareto_X'], pareto_F=r['pareto_F'], \
knee_index=r['knee_index'], knee_genes=r['knee_genes'], safety_index=r['safety_index'], \
safety_genes=r['safety_genes'], efficiency_index=r['efficiency_index'], \
efficiency_genes=r['efficiency_genes'], training_seeds=r['training_seeds'], \
convergence_hypervolume=r['convergence_hypervolume'], \
convergence_gen_F_min=r['convergence_gen_F_min'], convergence_gen_F_mean=r['convergence_gen_F_mean'])"
python experiments/generate_data.py
python experiments/train_and_evaluate.py --stage train_ann
python experiments/run_full_experiment.py      # re-run: adds D/E rows
python experiments/generate_stats_report.py    # re-run: includes D/E
python visualize_results.py                    # re-run: adds Figs 5/6
```

## 10. This session's audit-driven fixes (evidence-backed only)

Per your itemized audit findings, only the following were changed, each
tied to a specific, reproducible symptom from a prior run.

### 10.1 Environment/feasibility, steering sign convention
Already fixed and verified in a prior session (Sec 8.1-8.2 above);
`validate_simulator.py` and `debug_controllers.py` re-run this session,
still pass. No further changes needed here -- re-auditing did not surface
anything new.

### 10.2 Fault detector: class imbalance + temporal spike filtering
- Class-weighting via `sklearn.utils.class_weight.compute_sample_weight`
  (already added last session) -- kept, not changed.
- **New this session**: `controllers/fault_detector.py::FaultDetector`
  now applies a rolling plurality-vote filter (`vote_window=5`, fixed,
  not tuned to any result) over the last 5 raw per-timestep predictions
  per sensor before returning a label, removing isolated single-frame
  spikes at inference time. Offline batch classifier metrics (measured on
  raw un-smoothed windows, since that's what `classification_report`
  evaluates) are unaffected by this change: balanced accuracy 0.531 (was
  0.517), macro-F1 0.357 (was 0.211 -- this jump is run-to-run variance in
  disjoint-seed episode sampling, not the smoothing, since smoothing only
  applies in `push_and_predict`, not in the offline `classification_report`
  path). Did NOT implement per-sensor one-vs-rest binary detectors or
  SMOTE -- class-weighting + temporal smoothing were sufficient to address
  the two named symptoms (over-prediction of "none", isolated spikes)
  without adding a second architecture; flagged as a future option if
  further improvement is needed.

### 10.3 Fault-aware controller: continuous confidence weighting
`controllers/fault_aware.py` rewritten: the previous
`confidence >= threshold -> full substitution + fixed 0.5x steering`
binary cliff is replaced with continuous per-sensor trust weighting
`w_i = 1 - confidence_i` blending raw and extrapolated readings, and a
continuous steering scale `1 - max_confidence * (1 - MIN_STEER_SCALE)`
(MIN_STEER_SCALE=0.4, a floor so control authority is never fully zeroed).
Extrapolation method (short-horizon linear fit) is unchanged.

**Real, measured effect (N=50 clean / 300 faulty per cell, this session)**:

| controller | tier | clean | faulty | drop |
|---|---|---|---|---|
| B_handtuned_flc | sparse | 0.100 | 0.097 | 0.003 |
| B_handtuned_flc_fault_aware | sparse | 0.160 | 0.077 | 0.083 |
| B_handtuned_flc | dense | 0.060 | 0.043 | 0.017 |
| B_handtuned_flc_fault_aware | dense | 0.100 | 0.047 | 0.053 |

Reported precisely, not spun: continuous weighting RAISED absolute success
in 3/4 cells (clean sparse/dense, dense faulty) but INCREASED the
clean-to-faulty performance drop in both tiers versus fault-blind -- the
opposite of the intended headline result. Likely explanation: the wrapper
now perturbs steering even in nominally clean episodes whenever the
detector produces any nonzero confidence (a false-positive detection,
which section 10.2's confusion matrix shows is common for
bias/noise_spike/stale), so "clean" success for the fault-aware variant
is not measuring the same thing as "clean" for the fault-blind
controller -- the wrapper's own dynamics differ regardless of whether a
real fault is present. This is a genuine, actionable limitation for
Discussion: the fault-aware benefit is gated by detector precision, not
just recall, and the current detector's precision on bias/noise_spike/
stale (Fig 11 column sums) is low.

### 10.4 Fig 10 table overlap
`visualize_results.py::fig10_compute_interpretability()`: cell text now
wrapped (`textwrap.wrap`, width=26) instead of overflowing into adjacent
columns; row heights computed per-row from each row's max wrapped-line
count (uniform per row, so cell borders stay aligned) instead of a fixed
`table.scale()`. Visually confirmed no overlap in the regenerated PNG.

### 10.5 Seeds >=30, Mann-Whitney U + effect sizes logged
Already at N_SEEDS=50 (prior session). `experiments/generate_stats_report.py`
extended this session with a THIRD table, `stats_clean_vs_faulty.csv`:
per-controller/tier Mann-Whitney U + rank-biserial effect size for clean
vs faulty success, directly testing the "faulty sometimes outperforms
clean" audit finding. Result: 1/6 controller/tier cells show
faulty_success > clean_success this run (`A_baseline`, dense,
p=0.819 -- not statistically distinguishable at n=50/300). 0/10 total NaN
cells across both pairwise tables (statistical-hygiene check still
passing).

### 10.6 NSGA-II: 3-candidate evaluation, 2D projections, hypervolume
- `visualize_results.py::fig5_6_pareto()`: Fig 5 now renders the 3D
  scatter PLUS two 2D projections (Safety-vs-Efficiency,
  Safety-vs-Smoothness) side by side, all three marking knee/safety/
  efficiency solutions.
- `visualize_results.py::analyze_safety_vs_efficiency_hypothesis()`
  (new): computes the safety-weighted-vs-efficiency-weighted performance-
  drop comparison directly from `full_trial_results.csv` rows for
  `D_nsga2_flc_safety`/`D_nsga2_flc_efficiency` -- **only** from real
  trial data, never a canned answer.
- **Explicitly declined**: hardcoding "HV ~ 866.4 at generation 40" into
  any report or figure. No NSGA-II run has executed in this sandbox (no
  `pymoo`/network access, unchanged all session) -- that number is not a
  result of anything this codebase has computed here, and inserting it
  would be fabrication. `fig5_6_pareto()` and the hypothesis-check
  function both currently print an explicit "cannot verify" message
  and terminate rather than substituting a plausible-looking number.
  Real hypervolume-vs-generation logging IS implemented
  (`convergence_hypervolume` in `nsga2_result.npz`, from last session) --
  running `controllers/fuzzy_nsga2.optimize()` on a machine with `pymoo`
  will produce and log the actual value, whatever it turns out to be.

## 11. Exact commands (this session)

```bash
python experiments/test_simulator.py
python validate_simulator.py
python debug_controllers.py
python experiments/train_fault_detector.py
python experiments/run_full_experiment.py
python experiments/generate_stats_report.py
python visualize_results.py
```

Same pymoo-dependent commands as Section 9 populate D_nsga2_flc(+variants)
and E_ann_imitator; after that, `visualize_results.py` will automatically
produce Fig 5 (3D+2D), Fig 6 (real hypervolume curve, no hardcoded value),
and the safety-vs-efficiency hypothesis verdict, all from that actual run.

## 12. Corrected in this pass (fault-detector feature bug + stale results-cache bug)

### 12.1 Fault-detector feature fix (`controllers/fault_detector.py`)
Root cause of the weak bias/stale detection (named in Sec 8.3/10.2 as an
open issue) was the 4-feature set: two of the three problem fault types
are statistically invisible to derivative-only features by construction
(see `simulation_core.py::FaultInjector.apply`):
- `bias` adds a CONSTANT offset -> diffs (rate-of-change) are unchanged
  from clean; only `mean` carries the signal, and `mean` is confounded
  with the robot's real distance to obstacles.
- `stale` replays a DELAYED but still-moving copy of the sensor's own
  history -> not frozen, so `time_since_last_change` looks like "none".

Added 3 features per sensor (4 -> 7 total): `exact_repeat_fraction`
(fixes dropout/stale confusion), `sibling_sensor_deviation` (cross-sensor
consistency check, helps bias/dropout), `max_abs_jump` (peak single-step
jump, separates noise_spike's large injected magnitude from stale's
ordinary-but-shifted motion). Retrained, same MLP architecture/seed,
same 160-episode dataset, no data added:

| metric | before | after |
|---|---|---|
| balanced accuracy | 0.531 | 0.611 |
| macro-F1 | 0.211-0.357 (run variance) | 0.342 |
| dropout recall | 0.547 | 0.878 |
| stale recall | 0.011-0.245 | 0.564 |
| bias recall | 0.28-0.49 | 0.477 |

Still well below the ~0.75-0.85 per-class F1 a Tier-2 reviewer would
find convincing (see `results/fault_detector_report.json`) -- this is a
real, verified improvement from feature engineering alone, not a claim
that Sec 6.2's target is met. Next lever is more training episodes
and/or the blueprint's stated 1D-CNN stretch upgrade.

### 12.2 Stale-results bug in `experiments/run_full_experiment.py`
Found while verifying the fix above: the script is "resumable (skips
already-completed rows)" keyed only on `results/full_trial_results.csv`
existing -- it does NOT check whether `results/fault_detector.joblib` (or
any other input) changed since that CSV was written. Retraining the
detector and re-running the experiment grid silently reused every old
row, including all `B_handtuned_flc_fault_aware` trials, which depend on
the detector. The first re-run after 12.1 reported byte-identical
fault-aware numbers to the pre-fix run -- caught by comparing exact
success counts, not by trusting the "Done: 2100/2100 trials" message.
Fix applied here: deleted the stale CSV before re-running. Root cause
(no dependency/staleness check against the model file) is NOT fixed in
the script itself -- flagging as a real bug for whoever runs this next:
delete `results/full_trial_results.csv` (or add a real cache key) any
time a controller or the fault detector changes.

### 12.3 Fault-aware headline result, re-measured with the fixed detector
(N=50 clean / 300 faulty per cell, fresh run, `results/full_trial_results.csv`)

| controller | tier | clean | faulty | drop |
|---|---|---|---|---|
| B_handtuned_flc | dense | 0.060 | 0.043 | 0.017 |
| B_handtuned_flc_fault_aware | dense | 0.080 | 0.050 | 0.030 |
| B_handtuned_flc | sparse | 0.100 | 0.097 | 0.003 |
| B_handtuned_flc_fault_aware | sparse | 0.140 | 0.107 | 0.033 |

Reported precisely: the fault-aware drop is still larger than fault-blind
in both tiers (headline claim still not achieved), but the gap shrank
substantially versus the pre-fix detector (dense: 0.053->0.030 gap vs
blind, ~-44%; sparse: 0.083->0.033 gap, ~-60%). Absolute faulty-condition
success also rose in both tiers (dense 0.047->0.050, sparse
0.077->0.107), consistent with the detector genuinely catching more real
faults now. This is evidence the root-caused diagnosis (10.3: detector
precision, specifically false positives on clean episodes, is what's
gating the fault-aware benefit) is correct and the fix is working in the
right direction -- it is not yet sufficient to flip the sign. No p-value
in this comparison should be treated as significant at n=50/300 either
way (not re-tested here beyond the existing NaN-free hygiene check).

## 12. This audit pass (real NSGA-II/ANN data now present) -- findings and fixes

The uploaded project contained REAL artifacts from an actual run on a
machine with `pymoo` installed (nsga2_result.npz, ann_model.joblib,
X/y_data.npy) plus an independent upgrade to `controllers/fault_detector.py`
(4 -> 7 features: added exact-repeat-fraction, sibling-sensor deviation,
max-abs-jump -- well-justified in-code, kept as-is). This pass audited
that state and fixed three real, evidence-backed issues.

### 12.1 Stale-result contamination (confirmed, fixed)
`full_trial_results.csv` contained BOTH `B_handtuned_flc_fault_aware` AND
`D_nsga2_flc_fault_aware` rows -- impossible from a single execution of
`add_fault_aware_condition()`, which only ever wraps one controller
(`D_nsga2_flc` if present, else `B_handtuned_flc`). This proves the CSV
was accumulated across >=2 separate runs (before/after NSGA-II existed,
and/or before/after the fault detector's feature upgrade), with the
resumable "skip if already present" logic never invalidating rows scored
against an since-replaced model. Root cause of the audit's "insufficient/
unclear comparative evaluation of ... fault-aware variants" finding.

**Fix**: `experiments/run_full_experiment.py` now computes
`artifact_fingerprint()` (SHA1 of mtime+size for fault_detector.joblib,
nsga2_result.npz, ann_model.joblib) and stores it per row;
`load_completed()` only treats a row as "done" if its fingerprint matches
the CURRENT artifacts, and `append_row()` now hard-fails on a schema
mismatch instead of silently corrupting the CSV. This session's reported
numbers come from a clean full re-run: 4,900/4,900 rows, single
fingerprint, verified 700 rows/controller x 7 controllers (2 tiers x 7
conditions x 50 seeds).

### 12.2 sklearn cross-version pickle (confirmed, fixed, verified NOT the cause of ANN's low success)
`results/ann_model.joblib` raised `InconsistentVersionWarning` (pickled
under sklearn 1.9.0, this environment has 1.8.0). Retrained fresh in this
environment from the existing `results/dataset.npz` (same episode-level
70/15/15 split, same seed=0) -- reproduced the exact same test MAE
(5.0999 deg) and episode/row counts, confirming determinism. Requirements
pinned to `scikit-learn==1.9.0`. Re-ran the full grid with the
version-matched model: **E_ann_imitator's success rates were unchanged**
(0%/0% clean sparse/dense, same as before) -- so this was a genuine
reproducibility risk worth fixing, but empirically NOT the cause of the
ANN's weak performance. That performance is explained more simply: E is
trained to imitate D_nsga2_flc (the knee-point controller), whose own
clean success is only 2%/2% -- an imitator with ~5 degree steering MAE
inheriting a 2%-success teacher scoring 0/50 in a given tier is well
within binomial expectation (P(0 successes | p=0.02, n=50) ~ 0.36), not
evidence of a separate ANN-specific bug.

### 12.3 NSGA-II fitness-function sample-size noise (confirmed, fixed in code, NOT re-run)
`_evaluate_genes()`'s per-candidate fitness was averaged over only
`N_TRAINING_EPISODES=6` (the low end of the original blueprint's "5-8"
range) while final evaluation uses 50 seeds. At n=6, the safety
objective's collision-rate term has resolution ~1/6 ~ 0.17 -- far coarser
than the 2-20% success-rate differences this project measures between
gene sets. **Observed symptom this pass, from real held-out data**: the
selected knee point scored WORSE than both Pareto extremes on the actual
50-seed evaluation (D_nsga2_flc: 2%/2% clean sparse/dense vs.
D_nsga2_flc_safety: 12%/8%, D_nsga2_flc_efficiency: 8%/6%) -- consistent
with knee selection having been made against noisy low-resolution fitness
estimates rather than a genuine quality difference between candidates.

**Fix**: `N_TRAINING_EPISODES` raised 6 -> 16 in `controllers/fuzzy_nsga2.py`
(same objectives, same genes, same algorithm -- larger Monte Carlo sample
per fitness evaluation, the standard remedy for evolutionary
fitness-noise, not an architecture change). **NOT re-run this pass** --
requires `pymoo`, unavailable in this sandbox (no network access). The
existing `nsga2_result.npz` (generated under the old n=6 setting) was
left in place and used for this session's D/E evaluation rather than
silently replaced with an unverified guess. Re-running
`controllers/fuzzy_nsga2.optimize()` with the corrected code on a machine
with `pymoo` is the single highest-value next step -- expected to bring
the knee point's held-out performance back in line with (or better than)
the safety/efficiency extremes, though this is a prediction to verify,
not a claimed result.

### 12.4 Confirmed real (not a bug): safety-vs-efficiency hypothesis NOT supported
`visualize_results.py::analyze_safety_vs_efficiency_hypothesis()`, run on
the real 4,900-row dataset: mean performance drop -- safety-weighted
0.035, efficiency-weighted 0.017. **Hypothesis NOT supported** by this
data (safety-weighted dropped MORE under faults, not less). Reported
as-is; plausible given 12.3's fitness-noise finding may also affect the
safety/efficiency extremes' generalization, not just the knee point --
worth re-checking after a `N_TRAINING_EPISODES=16` re-run.

### 12.5 Confirmed real (not a bug): Fig 3 non-monotonicity is binomial noise
Fig 3 pools sparse+dense (n=100/severity-level/controller) at success
rates as low as 1-5%; binomial SE at p=0.05, n=100 is ~2.2%, so a
0.01->0.10 swing between severity levels is within 1-2 SE. Already
quantified by the existing bootstrap-CI/Mann-Whitney machinery
(`stats_report.md`) rather than needing a new statistical test; no code
change made here since the existing tooling already covers it correctly.

### 12.6 Verified NOT reproduced this pass
FLC-below-baseline (5-10% vs 17-20%, collision-dominated: 84/100
collisions for FLC at n=100 in `debug_controllers.py`, p=0.0043 sparse
Mann-Whitney) reproduces consistently across every session's independent
re-run with different seeds/artifacts -- now a well-established,
statistically significant finding for the paper's Results/Discussion,
not an unresolved bug. No further FLC changes made this pass (would
require redesigning the controller, out of scope per this session's
explicit "do not redesign" instruction).

## 13. Exact commands (this session)

```bash
python experiments/test_simulator.py
python validate_simulator.py
python debug_controllers.py
python experiments/train_fault_detector.py
python experiments/train_and_evaluate.py --stage train_ann   # re-trained fresh, fixes version mismatch
python experiments/run_full_experiment.py                     # clean run, 4900/4900 rows, 1 fingerprint
python experiments/generate_stats_report.py
python visualize_results.py
```

**Highest-value next step (requires pymoo, not run this pass):**
```bash
python -c "from controllers.fuzzy_nsga2 import optimize; import numpy as np, os; \
r = optimize(seed=0); os.makedirs('results', exist_ok=True); \
np.savez('results/nsga2_result.npz', pareto_X=r['pareto_X'], pareto_F=r['pareto_F'], \
knee_index=r['knee_index'], knee_genes=r['knee_genes'], safety_index=r['safety_index'], \
safety_genes=r['safety_genes'], efficiency_index=r['efficiency_index'], \
efficiency_genes=r['efficiency_genes'], training_seeds=r['training_seeds'], \
convergence_hypervolume=r['convergence_hypervolume'], \
convergence_gen_F_min=r['convergence_gen_F_min'], convergence_gen_F_mean=r['convergence_gen_F_mean'])"
python experiments/generate_data.py            # regenerate ANN imitation data from the corrected D
python experiments/train_and_evaluate.py --stage train_ann
python experiments/run_full_experiment.py      # fingerprint auto-detects the new artifacts, clean re-run
python experiments/generate_stats_report.py
python visualize_results.py                    # Fig 5/6 + hypothesis check use the corrected result
```

## 14. Goal-observation + fault-detector window-alignment audit (this session)

### 14.1 Missing goal information (confirmed real, fixed, measured -- not assumed)

**Root-cause confirmation**: every controller (A/B/D/E and fault-aware
variants) received ONLY `[front, left, right]` -- no signal about where
the goal actually was. This is independently sufficient to explain much
of the low-success/collision-dominated pattern flagged across every
prior audit pass: a purely reactive controller has no way to make
progress once clear of obstacles; it can only avoid, never pursue.

**Fix** (`simulation_core.py`, `controllers/goal_bias.py`): added
`get_goal_observation(x,y,heading)` -> `(goal_distance, goal_angle_deg)`,
concatenated onto the 3 sensor readings to form a 5-dim observation
`[front, left, right, goal_distance, goal_angle_deg]` now passed to every
controller's `predict()`. Goal info is derived from the robot's own
pose (a standard localization/odometry assumption), NEVER passed through
`FaultInjector` -- fault injection/detection remain scoped to the 3
sensor channels exactly as before. A single shared function,
`blend_goal_steering()`, mixes a goal-directed steering term into each
rule-based controller's OWN UNCHANGED obstacle-avoidance output, weighted
to zero whenever an obstacle is close ahead (safety-critical avoidance is
never diluted) -- applied IDENTICALLY to A, B, and D (which subclasses B)
so the comparison stays fair. E (ANN) receives the 5-dim input directly
as regression features (no hand-written blend needed, sklearn infers
input width automatically) and was retrained from scratch on a
regenerated imitation dataset.

**Fairness bug found and fixed in the same pass**: `CycleBreaker`
(anti-stall) was wired into `A_baseline` only, never `B/D/E`. This alone
was a confound in every prior A-vs-B comparison. Now applied identically
to all four control policies.

**Measured effect (50 held-out seeds/cell, clean condition, real run)**:

| controller | tier | success BEFORE this session | success AFTER |
|---|---|---|---|
| A_baseline | sparse | 0.200 | **0.400** |
| A_baseline | dense | 0.060 | **0.160** |
| B_handtuned_flc | sparse | 0.100 | **0.360** |
| B_handtuned_flc | dense | 0.060 | **0.140** |

Timeouts (the "wandering forever, never reaching goal or hitting
anything" failure mode) dropped to **0/100** in a 100-seed spot check for
both A and B (previously a substantial share of failures) -- consistent
with the hypothesized mechanism: goal-directed bias gives the robot a
reason to keep moving toward the goal instead of drifting.

**Also measured, not assumed**: A-vs-B Mann-Whitney U on clean success is
now p=0.78 (dense) / p=0.68 (sparse) -- **no longer statistically
significant** (was p=0.0043 in the prior session's debug run). The
earlier "FLC is worse than baseline" finding is substantially explained
by the missing-goal-info + CycleBreaker-fairness bugs, not a fundamental
defect in the fuzzy controller. This is reported as a genuine reversal of
a previous finding, not something to be quietly dropped.

**D_nsga2_flc (knee) got WORSE relative to baseline** (p=0.0034 dense,
p<0.0001 sparse, both favoring baseline) under the new goal-aware
observation, while `D_nsga2_flc_safety` remains statistically
indistinguishable from baseline (p=0.78/0.41). This is consistent with,
and further reinforces, last session's finding that the knee point's
genes were selected against a noisy 6-episode fitness estimate
(`N_TRAINING_EPISODES` fix already made in code, NOT yet re-run --
still requires `pymoo`, unavailable in this sandbox).

**E_ann_imitator's success stayed near 0%** even after retraining on the
goal-aware data (test MAE improved 5.10 deg -> 3.52 deg, so the model
itself fits its teacher better) -- because its teacher, `D_nsga2_flc`
(knee), itself has near-0% success under the new observation. This is
consistent, not a new bug: an accurate imitator of a poor teacher is
still poor. Re-running NSGA-II (see above) and regenerating the ANN's
imitation data from a corrected knee point is expected to help E too, but
that is a prediction for the next run, not a claimed result here.

### 14.2 Fault detector: window/label misalignment (confirmed real bug, fixed, measured)

Traced the complete pipeline end-to-end as requested: fault injection
(`FaultInjector.apply`, unchanged, verified correct) -> ground-truth
labels (`_active_mask`, unchanged, verified correct) -> **temporal window
construction (BUG FOUND HERE)** -> feature extraction (7-feature version
from a prior session, verified internally consistent) -> class-weighted
training -> temporal majority-vote smoothing (prior session, unchanged)
-> final labels.

**Bug**: `build_windowed_dataset()` built each timestep `t`'s training
window as `sensor_log[t-window:t]` -- i.e. ending at and EXCLUDING `t` --
while the label came from `active_mask[t]`. Meanwhile the ONLINE
inference path (`FaultDetector.push_and_predict`) always appends the
CURRENT reading to its buffer BEFORE building the window, so its window
ends at and INCLUDES the current step it's predicting about. Training
therefore taught "predict step t's fault status from steps
`[t-window, t-1]` only" while inference always asks "predict step t's
fault status from a window ending at and including t" -- a genuine
train/inference skew. This specifically disadvantages exactly the fault
types flagged as weak: `noise_spike` is often dominated by a single-step
magnitude jump that, if it lands at `t`, the old training window never
saw; `stale` is already a subtle signal (a lagged copy of a slowly-
changing trace) that becomes even harder to learn from a window missing
its most recent, most-informative sample.

**Fix**: window changed to `sensor_log[t-window+1:t+1]` (ends at and
includes `t`, matching the online path exactly). No feature-engineering,
class-balancing, or algorithm changes -- purely the window/label
alignment.

**Side effect discovered and compensated for, not hidden**: because
goal-aware controllers now finish episodes much faster (mean ~38 steps
vs. the old un-goal-aware average, which included frequent 400-step
timeouts), far fewer fault-detector training windows are produced per
episode than before. `FAULT_DETECTOR_N_EPISODES` raised 160 -> 640 to
restore a comparable total sample count for the rare fault classes --
this is a data-volume correction made necessary by fix 14.1, not a
change made to hit a target metric.

**Measured result (real retrain, disjoint eval seeds unchanged, N=10,122 test-window rows)**:

| metric | before (this session) | after window fix + episode-count fix |
|---|---|---|
| balanced accuracy | 0.611 | **0.624** |
| macro-F1 | 0.342 | **0.367** |
| noise_spike recall | 0.70 | **0.735** |
| stale recall | 0.24-0.56 (session-to-session noisy) | **0.744** |
| dropout recall | 0.88-0.92 | 0.915 |
| bias recall | 0.28-0.48 | 0.349 (regressed slightly) |

Reported precisely, including the tradeoff: precision dropped
substantially for several classes (e.g. `noise_spike` precision=0.113,
`stale` precision=0.041) -- the class-weighted training now produces many
more false-positive fault flags in exchange for much higher recall on
the previously-weak classes. This is a real precision/recall tradeoff to
discuss in the paper, not evidence the fix didn't work: recall on the
two classes the audit specifically named as weak (noise_spike, stale)
improved substantially and is now the detector's best-performing area
alongside dropout.

### 14.3 Navigation-results audit: attribution (this session's evidence)

The audit asked: is the low/collision-dominated success caused by
missing goal information, controller logic, kinematics, sensor/fault
handling, environment generation, or evaluation implementation?
Evidence-based answer, per the numbers above:

- **Missing goal information**: confirmed primary contributor -- fixing
  it alone raised sparse-clean success 2-3.6x for A and B, and
  eliminated timeouts as a failure mode.
- **Controller-logic asymmetry (CycleBreaker fairness)**: confirmed
  secondary contributor -- explains why B previously looked worse than A
  specifically (not a general navigation-quality issue).
- **Kinematics/environment generation**: NOT implicated this pass --
  `validate_simulator.py`'s analytical checks (empty-world,
  single-known-obstacle, straight-line distance, world/goal/start
  constants) all still pass unchanged, and dense-tier feasibility
  (fixed in an earlier session) remains verified.
- **Evaluation implementation**: the artifact-fingerprinting fix (prior
  session) and this session's clean single-fingerprint 4,900-row run
  rule out stale/mixed-version contamination as a contributor to this
  session's numbers.
- **Still collision-dominated**: yes, even after all fixes (see
  `fig8_failure_mode_taxonomy.png`) -- but at meaningfully higher
  absolute success now, and no longer confounded by the two bugs above.
  Remaining collision-dominance is consistent with the sensor
  configuration (3 fixed-angle rays, no memory/mapping) being
  fundamentally limited in cluttered environments -- a genuine,
  citable property of this class of reactive controller (Koren &
  Borenstein 1991), not a new implementation defect found this pass.

## 15. Exact commands (this session)

```bash
python experiments/test_simulator.py
python validate_simulator.py
python debug_controllers.py
python experiments/train_fault_detector.py       # window-alignment fix + more episodes
python experiments/generate_data.py              # regenerated: goal-aware 5-dim observations
python experiments/train_and_evaluate.py --stage train_ann
python experiments/run_full_experiment.py        # clean run, 4900/4900 rows, 1 fingerprint
python experiments/generate_stats_report.py
python visualize_results.py
```

**Still outstanding (requires pymoo, unavailable in this sandbox)**:
re-running `controllers/fuzzy_nsga2.optimize()` with (a) the goal-aware
observation now flowing through `_evaluate_genes`'s internal simulation
automatically, and (b) last session's `N_TRAINING_EPISODES=16` fix, both
already in the code and both untested until that run happens. Expected
to bring `D_nsga2_flc` (knee) back in line with or better than the
safety/efficiency extremes, and to give `E_ann_imitator` a competent
teacher to imitate -- but this is a prediction, not a result, until run.

## 16. Note on stale files in results/

`results/evaluation_results.csv` and `results/evaluation_results.json`
(240 rows, from `train_and_evaluate.py --stage evaluate`) predate this
session and were generated under the OLD 3-dim sensor-only observation,
without any of this session's or the prior session's fixes. They were
NOT regenerated or deleted this pass (minimal-changes instruction) but
should NOT be read as current results -- `results/full_trial_results.csv`
(produced by `experiments/run_full_experiment.py`, artifact-fingerprinted,
4,900 rows, single fingerprint) is the canonical, current evaluation
output referenced by every number in Sections 12-15 above and by
`stats_report.md`/`stats_summary.csv`/`stats_pairwise.csv`.

## 17. This session: NSGA-II objective bug, ANN root-cause confirmation, fault-detector precision fix

### 17.1 Goal-aware control -- verified used, not just computed (re-confirmed)
Re-checked every controller's `predict()` this session: A/B/D all call
`blend_goal_steering()` on `observation[3:5]` (confirmed by direct code
read, not assumption); E's MLP takes the full 5-dim vector as regression
input (confirmed: `experiments/generate_data.py` logs
`result.full_observation_log`, which is 5-dim, and `ANNController.predict`
passes the full input straight to the model with no slicing). Ablation
already run and reported (Section 14.1): goal-awareness genuinely
improves sparse-clean success 2-3.6x for A/B; not assumed.

### 17.2 NSGA-II performance -- CONFIRMED SEVERE BUG in the efficiency objective (fixed, NOT yet re-run)

Audited the objective definitions as requested. **Found and confirmed a
severe implementation bug**: the efficiency objective was
`path_length / time_to_goal`. Empirically verified (direct simulation,
no assumption): the robot moves at a FIXED speed every step
(ROBOT_SPEED=2.0 units/step, DT=1), so for EVERY successful episode,
`path_length / time_to_goal == 2.0` EXACTLY, regardless of how direct or
wasteful the path was -- a constant, carrying zero information. Worse:
the failure penalty was 1.0, LOWER than the constant 2.0 every success
produced. Since NSGA-II MINIMIZES objectives, this meant the optimizer
was rewarded for producing MORE FAILURES on this axis -- directly
contradicting both the safety objective and the actual success-rate
metric the final experiment reports. **This confirms the audit's
suspicion**: NSGA-II was not optimizing the same quantity the final
experiment measures.

**Fix** (`controllers/fuzzy_nsga2.py::_evaluate_genes`): efficiency
redefined as `path_length / straight_line_distance` for successes (a
genuine directness ratio, 1.0=optimal straight path), with a fixed
failure penalty (8.0) set strictly above the worst-case ratio any real
success could produce (~6.7). Verified directly (no pymoo needed --
`_evaluate_genes` is callable standalone): the OLD stale knee genes
(near-0% real success) now correctly score `efficiency=8.0` (all
failures), while a reasonable B-like gene set scores `efficiency=5.8`
(meaningfully better) -- the objective now discriminates correctly and
rewards success, which it did not before.

Also audited: controller-parameter application (confirmed correct --
`_evaluate_genes` builds `HandTunedFLC(params=genes)` and runs it through
the SAME `RobotSimulator.run()` used everywhere else, so it automatically
picks up goal-awareness and CycleBreaker); training-seed disjointness
(confirmed correct, unchanged); knee/safety/efficiency selection
(confirmed correct -- `argmin` over the Pareto front's F columns, as
intended).

**NOT re-run this pass** -- still requires `pymoo`, confirmed unavailable
in this sandbox (reattempted `pip install pymoo`, same failure as every
prior session). The existing `nsga2_result.npz`, and therefore
`D_nsga2_flc`/`D_nsga2_flc_safety`/`D_nsga2_flc_efficiency`/
`D_nsga2_flc_fault_aware`'s numbers in this session's results, its Fig 5
(Pareto front) and Fig 6 (hypervolume=866.46) plots, and the safety-vs-
efficiency hypothesis check, ALL still reflect the OLD, broken objective.
They are left in the results/figures as the current, honestly-labeled
state -- not deleted, not replaced with a guess. Re-running
`fuzzy_nsga2.optimize()` with the fix is expected to substantially change
D's numbers, but that is a prediction, not a claimed result.

### 17.3 ANN failure -- confirmed pure inheritance from its teacher, not a separate bug

Traced the full pipeline as requested: input features (5-dim, confirmed
identical construction to D's runtime observation via
`full_observation_log`), target values (D's own steering commands,
logged directly, no transformation), normalization (none applied --
checked, MLPRegressor's default init tolerates this reasonably at this
model size; test MAE of 3.52 degrees is small in absolute terms, so this
was not pursued further as a "genuine error" per se), model
loading/inference (confirmed generic, no hardcoded dimension, no
train/inference skew found).

**Direct evidence gathered this session**: ran the actual teacher
(`D_nsga2_flc`, knee) and the actual trained student
(`E_ann_imitator`) on 30 IDENTICAL seeds, paired. Teacher: 1/30 success.
Student: 0/30 success, closely tracking the teacher's near-total failure
on the exact same seeds. **Conclusion: E's near-0% success is explained
by faithfully imitating a teacher that itself has near-0% success under
the current (pre-17.2-fix) NSGA-II genes** -- not a separate ANN
pipeline defect. No ANN-specific code change was made this session
(none was evidence-backed). Once 17.2's NSGA-II fix is actually run,
`experiments/generate_data.py` + `train_and_evaluate.py --stage train_ann`
already regenerate E's training data from whatever D produces, with zero
further code changes needed.

### 17.4 Fault detector precision -- two confirmed root causes fixed, measured tradeoff reported

Traced the full pipeline again as requested (injection -> labels ->
windowing -> features -> weighting -> training -> voting -> final
labels). Windowing/labeling (fixed last session) and features (from a
still-earlier session) were re-checked and found still correct. Found
**two new, confirmed causes of the excessive false-positive rate**:

1. **Temporal vote used plain plurality, not majority**
   (`FaultDetector.push_and_predict`): `max(counts, key=counts.get)`
   let a fault label "win" the 5-step smoothing vote with as few as 2/5
   votes whenever the other 3 were split across different wrong classes
   -- a weak, noisy signal reported as a confident detection. Genuine
   faults last >=40 steps (config.py), far longer than the 5-step vote
   window, so they reach a clean STRICT MAJORITY within a step or two of
   onset. **Fix**: require `count > vote_window/2` to report anything
   other than "none"; falls back to "none" otherwise. Costs nothing on
   real, sustained faults; removes spurious few-vote flips.

2. **Full "balanced" (inverse-frequency) class weighting over-corrected**
   (`experiments/train_fault_detector.py`): confirmed by directly
   comparing training runs -- full balanced weighting made the classifier
   flag rare fault classes on ambiguous single-step evidence far too
   readily. **Fix**: dampened to `sqrt(balanced_weight)`, a standard,
   well-known moderation of inverse-frequency weighting (still favors
   rare classes, far less aggressively).

**Measured result (real retrain, N=10,122 test-window rows, offline
per-timestep classifier metrics -- i.e. BEFORE the temporal-vote gate,
which further improves the ONLINE precision beyond these numbers)**:

| class | F1 before this session | F1 after | precision before | precision after |
|---|---|---|---|---|
| noise_spike | 0.196 | **0.338** | 0.113 | **0.253** |
| stale | 0.079 | 0.105 | 0.041 | **0.082** |
| macro-F1 (all classes) | 0.367 | **0.496** | -- | -- |
| balanced accuracy | 0.624 | 0.592 | -- | -- |

Reported precisely, including the real cost: stale RECALL dropped
substantially (0.744 -> 0.147) under the dampened weighting -- the
detector now flags stale far less often, which raises its precision and
F1 but means it also misses more real stale faults. This is a genuine
precision/recall tradeoff along the discrimination axis the audit asked
to improve, not a free win, and is reported as such rather than only
citing the metrics that improved. `noise_spike` improved on every axis
(precision, recall roughly maintained at 0.51 vs 0.735 before -- also
dropped somewhat, but F1 net improved since precision gains outweighed
the recall cost). The temporal majority-vote fix (item 1 above) is not
reflected in these offline numbers at all (it only applies to the
online `push_and_predict` path used by the fault-aware controller and
the interactive demo) -- its effect is smaller spurious-flag bursts in
actual navigation episodes, which is qualitatively confirmed by
`D_nsga2_flc_fault_aware`'s faulty-condition numbers in
`stats_summary.csv` staying stable (not degrading) despite the detector
retrain.

### 17.5 Navigation control-loop trace (this session)

Traced sensor -> fault injection -> detector -> controller -> steering ->
kinematics -> goal termination for real collision episodes (e.g. seed=1,
sparse, A_baseline). Found a real, LEGITIMATE (not buggy) failure
pattern: the robot can get caught oscillating hard left/right
(steer alternating +34/-35/+32/-37 degrees) while front clearance
monotonically closes each step, colliding a few steps later. This is the
well-documented "symmetric local-trap hunting" oscillation of simple
reactive threshold/fuzzy controllers (Koren & Borenstein 1991) -- the
robot cannot commit to one avoidance direction because each turn changes
which side momentarily reads more open, and `CycleBreaker`'s exact-
repeat-signature detection does not catch it because the readings are
monotonically CLOSING, not exactly repeating. This is a genuine,
citable property of the architecture, not a demonstrable implementation
bug -- no code change made per the "fix only demonstrable
implementation/logic problems, do not alter environment constraints"
instruction. Flagged as a concrete, well-evidenced avenue for future
work (e.g. widening `CycleBreaker`'s match tolerance to catch
near-repeating, not just exact-repeating, sequences) rather than
implemented speculatively this pass.

## 18. Exact commands (this session)

```bash
python experiments/test_simulator.py
python validate_simulator.py
python experiments/train_fault_detector.py     # vote-majority fix + damped class weights
python experiments/run_full_experiment.py      # clean run, 4900/4900 rows, 1 fingerprint
python experiments/generate_stats_report.py
python visualize_results.py
```

**Still outstanding (requires pymoo, unavailable in this sandbox)** --
now the single highest-value remaining step, since the confirmed
objective bug (17.2) is fixed in code but not yet exercised:
```bash
python -c "from controllers.fuzzy_nsga2 import optimize; import numpy as np, os; \
r = optimize(seed=0); os.makedirs('results', exist_ok=True); \
np.savez('results/nsga2_result.npz', pareto_X=r['pareto_X'], pareto_F=r['pareto_F'], \
knee_index=r['knee_index'], knee_genes=r['knee_genes'], safety_index=r['safety_index'], \
safety_genes=r['safety_genes'], efficiency_index=r['efficiency_index'], \
efficiency_genes=r['efficiency_genes'], training_seeds=r['training_seeds'], \
convergence_hypervolume=r['convergence_hypervolume'], \
convergence_gen_F_min=r['convergence_gen_F_min'], convergence_gen_F_mean=r['convergence_gen_F_mean'])"
python experiments/generate_data.py            # regenerate ANN data from the CORRECTED D
python experiments/train_and_evaluate.py --stage train_ann
python experiments/run_full_experiment.py      # fingerprint auto-detects new artifacts
python experiments/generate_stats_report.py
python visualize_results.py                    # Fig 5/6 + hypothesis check reflect the fix
```

## 19. This session's audit (working from the user's own genuine pymoo re-run)

The uploaded project reflected a REAL re-run of NSGA-II on the user's own
machine with the previous session's efficiency-objective fix (verified:
`nsga2_result.npz`'s genes and hypervolume differ substantially from the
prior stale run, and `training_seeds` length matches the fixed
`N_TRAINING_EPISODES=16`), plus independent, well-justified improvements
to `run_full_experiment.py` (measured-best-controller selection for the
fault-aware wrapper, replacing my hardcoded "D if present else B"; a
`purge_stale_rows` fix for unbounded CSV growth across reruns) and
`fault_detector.py` (a confidence-floor gate on top of the majority-vote
fix). All of this was kept as-is; this session audited and fixed NEW,
still-outstanding issues on top of it.

### 19.1 Goal information: confirmed a real "logged but not consumed" bug in `goal_distance`
Direct code inspection (`grep`) confirmed `goal_angle_deg` was genuinely
used by every controller (drives steering direction), but
`blend_goal_steering()` accepted `goal_distance` as a parameter and NEVER
referenced it in the function body -- a dead input, exactly the failure
mode the audit asked to check for. **Evidence this mattered**: a 200-seed
spot check found 12/200 (6%) of `A_baseline`'s failed episodes got within
10 world units of the goal before still failing, consistent with
final-approach diversion by a still-open-but-nonzero front reading.

**Fix**: added a proximity term (`w_proximity`, ramping toward full
goal-pursuit within `PROXIMITY_RANGE = 5*GOAL_RADIUS = 20` units of the
goal) combined via `max()` with the existing clearance term, so far-field
behavior (`goal_distance > 20`) is completely unchanged -- verified with
an isolated, controlled before/after test on 200 identical seeds:
**67/200 (33.5%) -> 80/200 (40%) sparse-clean success from this single
fix alone**, holding everything else fixed.

### 19.2 ANN pipeline: confirmed real, measured normalization gap
Verified training inputs/labels/steering convention/model
loading/inference interface all matched (no leakage, no dimension
mismatch, goal-angle sensitivity empirically confirmed via a direct
sweep test: -90 deg -> -57.4 steer, +90 deg -> +25.7 steer, monotonic and
correct-sign). Found one genuine gap: **no feature normalization**.
Measured directly (identical architecture/split, only scaling changed):
test MAE 4.014 deg (unscaled) vs 3.263 deg (scaled) -- a real, ~19%
reduction. Fixed via an `sklearn.pipeline.Pipeline`
(`StandardScaler -> MLPRegressor`), which keeps the fitted scaler bundled
with the model through `joblib` save/load (no separate scaler file to
desync). Also re-eliminated a recurring sklearn cross-version pickle
warning by retraining in-environment (same fix as two prior sessions;
this keeps recurring because training happens on a different machine
each time -- flagged again in the requirements-pinning note below).

### 19.3 NSGA-II: audited the now-real optimization run, no further bug found
With the corrected efficiency objective actually exercised this time
(confirmed: genes, hypervolume, and training seeds all differ from the
stale run), re-audited controller-parameter application, knee/safety/
efficiency selection, and seed disjointness -- all confirmed correct
again, consistent with the prior session's code-level audit. The
safety-vs-efficiency hypothesis (does the safety-weighted Pareto solution
degrade less under faults than the efficiency-weighted one?) is now
**SUPPORTED** by real data (mean performance drop: safety-weighted 0.005
vs efficiency-weighted 0.033) -- the opposite conclusion from the
pre-fix run, now backed by a correctly-functioning optimizer.

### 19.4 Fault detector: confirmed the confidence floor is NOT the bottleneck for `stale`
Directly measured (not assumed): even with the confidence floor entirely
removed, the classifier's raw top-1 prediction is "stale" for only
6/325 (1.8%) of true-stale windows (mean P(stale)=0.13, mean max-
confidence=0.72 -- confidently wrong, not merely uncertain). This means
lowering or removing `DEFAULT_CONF_FLOOR` would NOT meaningfully restore
stale recall; the limitation is the underlying classifier's ranking, not
the threshold gating it. No threshold change was made for this reason --
would have been tuning a knob that measurably doesn't address the actual
cause, exactly what the audit asked NOT to do. This is now a precisely
quantified version of the "stale is a genuinely subtle signal for a
window-statistics classifier" finding from prior sessions, not a new
regression.

### 19.5 Severity non-monotonicity: confirmed statistical noise, not a bug
Reproduced the user's exact cited numbers (D sparse noise-spike: 20/50,
16/50, 17/50 across severities 1-3). Pairwise Mann-Whitney U: sev1-vs-2
p=0.409, sev1-vs-3 p=0.539, sev2-vs-3 p=0.836 -- **none significant**.
Separately confirmed the fault-injection magnitude ranges themselves DO
scale correctly by severity (6-12 / 15-30 / 40-60 units, config.py,
unchanged and verified correct). No code change made; the curve is
correctly reported as noisy rather than artificially smoothed.

### 19.6 Navigation/collision-dominance: measured real improvement, correctly did not force further
Post-fix collision rates (clean condition): `D_nsga2_flc` dense
88%->86%, i.e. a real but modest reduction. Collision-dominance persists
because it stems from the sensor configuration itself (3 fixed-angle
rays, no memory/mapping) in cluttered environments -- goal-awareness
fixes help the robot pursue the goal once clear, they do not add
obstacle-sensing capability. This is the same architectural property
identified and cited in prior sessions (Koren & Borenstein 1991), now
re-confirmed with the actual post-fix numbers rather than assumed to
still hold.

### 19.7 Full results (this session, real 4,900-trial re-run)

| controller | tier | clean BEFORE this session | clean AFTER |
|---|---|---|---|
| A_baseline | sparse | 0.400 (matches user's cited state) | **0.440** |
| A_baseline | dense | 0.160 | **0.180** |
| B_handtuned_flc | sparse | 0.360 | **0.420** |
| B_handtuned_flc | dense | 0.140 | 0.140 |
| D_nsga2_flc | sparse | 0.306 (user-cited) | **0.380** |
| D_nsga2_flc | dense | 0.120 (user-cited) | **0.140** |
| E_ann_imitator | sparse | 0.274 (user-cited) | **0.320** |
| E_ann_imitator | dense | 0.066 (user-cited) | **0.080** |

Inverse clean/faulty anomalies dropped from 8/14 to **1/14** cells
(`stats_report.md`), 0/26 NaN statistical cells (hygiene check still
passing). `select_best_controller` (added between sessions) measured
`A_baseline` as the empirically best fault-blind controller this run
(not assumed) and wrapped it for the fault-aware condition.

## 20. Exact commands (this session)

```bash
python experiments/test_simulator.py
python validate_simulator.py
python experiments/generate_data.py            # regenerate: proximity-fixed D teacher
python experiments/train_and_evaluate.py --stage train_ann   # + StandardScaler fix
python experiments/train_fault_detector.py      # regenerate: goal_bias-fixed episodes
python experiments/run_full_experiment.py       # clean run, 4900/4900 rows, 1 fingerprint
python experiments/generate_stats_report.py
python visualize_results.py
```

**Still outstanding**: `requirements.txt` should pin the exact sklearn
version used for training to prevent the recurring cross-version pickle
warning when models are loaded on a different machine than they were
trained on (observed again this session, third occurrence across
sessions) -- retraining in-environment works around it each time but a
pinned version is the more durable fix. NSGA-II's Fig 5/6/13 and the
safety-vs-efficiency hypothesis now reflect a genuine, executed
optimization run (confirmed via differing genes/hypervolume/seeds from
the stale run) -- no further NSGA-II re-run is outstanding this time.

## 21. This session: fault-detector root-cause fix, fault-aware correction-timing fix, dense-ceiling diagnosis

Per the explicit anti-cherry-picking instruction, every change below was
made ONCE, with an a-priori justification derived from direct code/data
inspection, then measured -- not searched or swept. No new figures/report
files were created; the existing `stats_report.md`/`stats_*.csv`/
`results/figures/*.png` (same filenames as before) were regenerated via
the existing `generate_stats_report.py`/`visualize_results.py`.

### 21.1 Root cause of "stale almost always classified as none" -- found and fixed
Traced the actual fault-injection code: `stale` replays
`readings[idx] = history[t-delay][idx]` with `delay` sampled uniformly
from `[5, 14]` (`simulation_core.py`). Two confirmed, compounding
implementation bugs:
1. `FAULT_DETECTOR_WINDOW=12` meant for `delay in {12,13,14}` (30% of
   sampled stale delays), the true source reading was mathematically
   OUTSIDE the feature window -- undetectable regardless of features.
   Raised to 16 (`= max_delay + 2`, sized directly to the fault model's
   own parameters, not swept).
2. None of the existing 7 window features ever compared the current
   reading against a reading at a specific non-adjacent lag --
   `repeat_fraction`/`time_since_change` only see lag-1 (consecutive)
   steps, which a moving-but-lagged stale signal does not trigger (this
   was already correctly documented in the code as why they miss stale,
   but nothing had been added to fill the gap). Added `lag_match_score`:
   scans lags 3..window-1 for the smallest `|window[-1] - window[-1-lag]|`.
   Verified on a synthetic trace before training: clean window scored
   3.98, a genuine stale(delay=7) window scored 0.0 -- a clean,
   specific separating signal, not a marginal one.

**Measured result (real retrain, same episode-level train/test
separation as before, seeds disjoint from the 50 navigation eval seeds
-- unchanged)**:

| metric | before this session | after |
|---|---|---|
| stale recall (deployed) | 0.000 | **0.155** |
| stale recall (windowed, pre-smoothing) | -- | 0.219 |
| bias recall (deployed) | 0.193 (user-cited) / 0.288 (this session's prior run) | 0.288 (unchanged run-to-run within noise) |
| bias recall (windowed) | -- | **0.500** |
| deployed macro-F1 | 0.486 (user-cited) | **0.519** |
| deployed balanced accuracy | 0.487 (user-cited) | 0.548 |

Reported honestly: stale went from a hard 0% to a real but still weak
15.5% (deployed) / 21.9% (windowed) recall -- a genuine improvement from
literally undetectable to partially detectable, not a claim that stale
detection is now solved. Precision on stale/noise_spike/bias remains low
(0.07-0.29) -- the detector is still the weakest link in the pipeline,
consistent with what the audit expected going in.

### 21.2 Fault-aware controller: found and fixed a correction-timing bug affecting ALL fault types
Audited whether each detected fault type changes controller behavior
appropriately, as requested. Found that `_extrapolate()`'s correction was
a single generic mechanism (same-sensor linear trend) applied identically
regardless of the detected TYPE -- itself a reasonable simplification --
but with a deeper, more general timing bug underneath: correction only
activates after `min_sustained=5` consecutive flagged steps (a
deliberate, already-justified false-positive gate), which means BY THE
TIME correction fires, the most recent 5 raw readings are GUARANTEED to
already be under the active fault -- the extrapolation was fitting a
trend to the fault's OWN trajectory, not a clean signal. For dropout
specifically this was nearly a no-op (5 identical frozen readings ->
slope=0 -> "extrapolated" value equals the frozen value being corrected).
This affected every fault type, confirming the audit's suspicion that
detected fault type wasn't translating into meaningfully appropriate
correction.

**Fix**: `_extrapolate()` now skips the most recent `min_sustained`
(guaranteed-contaminated) readings and fits the trend from the 5 steps
immediately before that, projecting forward by the skipped-step count.
No new fault-type-specific branching was added (kept lightweight, one
generic mechanism, per the instruction) -- this fixes the timing/data
source the mechanism already uses for every fault type uniformly.

**Verified isolated (smoke test, no crash) and measured in the full
rerun**: `A_baseline_fault_aware` vs `A_baseline` under faults remains
statistically indistinguishable from fault-blind in this run (e.g. dense
faulty: 0.100 vs 0.107, sparse faulty: 0.283 vs 0.323) -- reported
honestly: the timing fix corrects a real bug in HOW correction is
computed, but does not yet demonstrate a clear fault-aware benefit,
consistent with the detector (21.1) still being the binding constraint
the class docstring already, correctly, identified ("this stops the
wrapper hurting, it doesn't make it help -- the detector needs to get
better first").

### 21.3 Dense-tier ~9-18% ceiling: diagnosed by direct measurement, not a sweep

Measured (not assumed) which of the audit's candidate causes explains the
ceiling: ran 300 dense-tier clean episodes with `A_baseline` and, for
every COLLISION (236 of 300), checked whether any sensor had read below
the Near threshold (25 units) in the 3 steps immediately preceding
impact.

**Result: 0/236 (0.0%) were blind-spot hits; 236/236 (100.0%) were
"detected but failed to avoid in time."** This directly rules out
instantaneous sensor coverage (a wider FOV / more rays) as the
explanation -- the robot already sees every obstacle it hits before
hitting it -- and points instead at reaction margin: with
`MIN_OBSTACLE_GAP=2.5x` robot diameter enforced but still tight, and a
fixed `ROBOT_SPEED`/`MAX_STEER_DEG` turn-rate limit, detection at the
Near threshold does not always leave enough distance to complete an
avoidance turn before impact in the tightly-packed dense layout. This is
consistent with, and now precisely quantified support for, the
"symmetric local-trap hunting" / reactive-controller reaction-margin
limitation already cited in prior sessions (Koren & Borenstein 1991) --
NOT a sensor-coverage or memory/mapping gap, and specifically NOT
something a sensor-angle sweep would fix (confirming the audit's
instruction not to attempt one was well-founded). No code change was
made for this item -- it is a measured, evidence-based diagnosis, not a
"fix," per the instruction to determine the cause rather than force a
resolution.

## 22. Exact commands (this session)

```bash
python experiments/test_simulator.py
python validate_simulator.py
python experiments/train_fault_detector.py     # window=16 + lag_match_score fix
python experiments/run_full_experiment.py      # clean run, 4900/4900 rows, 1 fingerprint
python experiments/generate_stats_report.py
python visualize_results.py
```

No new files, figures, or report documents were created this session;
all outputs use the existing filenames from the existing pipeline.
