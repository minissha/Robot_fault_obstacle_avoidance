"""
train_fault_detector.py

Blueprint v2 Section 6.2: generate a rolling-window fault-detection
dataset for FREE from the existing fault-injection module (every episode
already knows its ground-truth fault type/timing), train the lightweight
MLP classifier, and report accuracy/precision/recall/F1 per class plus a
confusion matrix (Figure 11).

Episode-level train/test split (not per-timestep) to avoid leakage, per
the blueprint.

Run from the project root:
    python experiments/train_fault_detector.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from config import (
    FAULT_DETECTOR_CLASSES, FAULT_DETECTOR_N_EPISODES, FAULT_DETECTOR_PATH,
    FAULT_DETECTOR_WINDOW, RESULTS_DIR,
)
from simulation_core import RobotSimulator, sample_fault_spec_typed, FAULT_TYPES
from controllers.baseline import BaselineController
from controllers.fuzzy_handtuned import HandTunedFLC
from controllers.fault_detector import (
    build_windowed_dataset, build_classifier, FaultDetector, DEFAULT_CONF_FLOOR,
)


# How far the faulted reading has to sit from the true one before the fault
# counts as visible. Baseline sensor noise is 0.5 units, so 2.0 is about
# four standard deviations -- comfortably outside what clean noise produces.
OBSERVABILITY_THRESHOLD: float = 2.0

# Which fault (or none) each episode carries, cycled over the run.
#
# It is not an even split, and it is not meant to be. One ray out of eleven
# is faulty at a time, so a faulty episode is still mostly healthy readings,
# and the fault types do not all last as long: a bias runs to the end of the
# episode while a stale fault covers about a third of it. Left on an even
# split there were roughly 130 healthy windows for every stale one, and the
# classifier did the sensible thing and called them all healthy. Giving the
# short-lived faults more episodes evens out the number of windows each one
# actually contributes.
EPISODE_PLAN = (
    None, "stale", "noise_spike", None, "stale", "dropout",
    "noise_spike", "stale", "bias", None, "noise_spike", "dropout",
)

# Cap on how many "none" windows go into training, as a multiple of the
# largest fault class. Training on the raw ratio drowns the fault classes
# out. The held-out set is left exactly as it is, so the reported numbers
# still describe the real distribution.
NONE_TRAIN_RATIO: float = 12.0

# Clean readings the detector is allowed to flag, as a fraction. The
# confidence floor is picked on held-out episodes as the strictest setting
# that still stays inside this budget -- rather than being a number typed
# into the source and left there. False alarms are not free: the
# fault-aware controller acts on every one of them.
FALSE_ALARM_BUDGET: float = 0.04

CONF_FLOOR_GRID = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _active_mask(spec, n_steps: int, observed_log=None, clean_log=None,
                  window: int = FAULT_DETECTOR_WINDOW):
    """Steps where the fault was running AND actually changing the reading.

    The second half of that matters more than it sounds. A stale fault on a
    ray staring into open space replays a reading of 100 as a reading of
    100: the fault is running, the label says "stale", and there is nothing
    in the numbers to find. Marking those steps as faulty asks the detector
    to guess, and then scores it for guessing wrong -- which is where most
    of the old stale recall of 0.15 was going.

    A fault counts as visible at step t if it moved the reading by more than
    OBSERVABILITY_THRESHOLD anywhere in the window ending at t, since the
    window is what the detector actually gets to look at.
    """
    mask = [False] * n_steps
    if spec is None:
        return mask

    running = [spec.start_step <= t < spec.start_step + spec.duration
               for t in range(n_steps)]
    if observed_log is None or clean_log is None:
        return running

    obs = np.asarray(observed_log, dtype=np.float64)
    clean = np.asarray(clean_log, dtype=np.float64)
    delta = np.abs(obs[:, spec.sensor_index] - clean[:, spec.sensor_index])
    visible = delta > OBSERVABILITY_THRESHOLD

    for t in range(n_steps):
        if not running[t]:
            continue
        lo = max(0, t - window + 1)
        mask[t] = bool(visible[lo:t + 1].any())
    return mask


def score(y_true, y_pred, title: str) -> dict:
    """Print the usual metrics for one way of scoring the detector.

    Shared by both scoring passes below, so any difference between them is
    down to the detector and not to how it was measured.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels_present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels_present, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels_present,
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    cm_row_normalized = cm.astype(float) / np.maximum(1, cm.sum(axis=1, keepdims=True))

    print(f"\n--- {title} ---")
    print(f"  rows scored: {len(y_true)}")
    print(f"  accuracy (majority-dominated, reference only): {acc:.3f}")
    print(f"  BALANCED accuracy (mean per-class recall):     {balanced_acc:.3f}")
    print(f"  macro-F1 (unweighted mean of per-class F1):    {macro_f1:.3f}")
    print("  per-class precision/recall/F1:")
    for lbl in labels_present:
        r = report[lbl]
        print(f"    {lbl:<12s} precision={r['precision']:.3f} recall={r['recall']:.3f} "
              f"f1={r['f1-score']:.3f} support={int(r['support'])}")
    print("  confusion matrix (rows=true, cols=pred, raw counts):")
    print(f"  {labels_present}")
    for row in cm:
        print("   ", row)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "labels": labels_present,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_normalized": cm_row_normalized.tolist(),
        "classification_report": report,
        "n_rows": int(len(y_true)),
    }


def evaluate_deployed(model, episodes, test_eps, window: int,
                       conf_floor: float = DEFAULT_CONF_FLOOR):
    """
    Score the detector the way it's actually used.

    The reported numbers used to come from calling the model directly on the
    test set, but nothing in the project uses it like that. The fault-aware
    controller goes through push_and_predict, which smooths over the last few
    predictions first. So the confusion matrix in the report and the detector
    the controller actually sees were two different things.

    This replays each test episode step by step through the real path. It
    scores the same steps the direct version does, so the two are comparable.
    """
    detector = FaultDetector(model=model, window=window, conf_floor=conf_floor)
    y_true, y_pred = [], []
    n_sensors = detector.n_sensors
    for ep in episodes:
        if ep["episode_id"] not in test_eps:
            continue
        spec = ep["fault_spec"]
        log = ep["sensor_log"]
        mask = _active_mask(spec, len(log), log, ep["clean_log"])
        detector.reset()
        for t, reading in enumerate(log):
            labels, _confs = detector.push_and_predict(np.asarray(reading, dtype=np.float64))
            if t < window - 1:
                continue        # buffer not yet full: the windowed view has no row here either
            for s in range(n_sensors):
                truth = (spec.fault_type
                         if spec is not None and s == spec.sensor_index and mask[t]
                         else "none")
                y_true.append(truth)
                y_pred.append(labels[s])
    return y_true, y_pred


def select_conf_floor(model, episodes, val_eps, window: int):
    """Pick the confidence floor on episodes the model never trained on.

    Sweeps the grid and keeps the setting with the best macro-F1 among those
    whose false-alarm rate on genuinely clean readings stays inside
    FALSE_ALARM_BUDGET. If nothing clears the budget, takes whichever
    setting has the lowest false-alarm rate -- so the choice degrades to
    "be as cautious as possible" rather than silently ignoring the budget.
    """
    rows = []
    for floor in CONF_FLOOR_GRID:
        y_true, y_pred = evaluate_deployed(model, episodes, val_eps, window,
                                            conf_floor=floor)
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        clean = y_true == "none"
        false_alarm = float((y_pred[clean] != "none").mean()) if clean.any() else 0.0
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        macro_f1 = float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0))
        rows.append({"conf_floor": float(floor), "false_alarm_rate": false_alarm,
                      "macro_f1": macro_f1})

    within = [r for r in rows if r["false_alarm_rate"] <= FALSE_ALARM_BUDGET]
    best = (max(within, key=lambda r: r["macro_f1"]) if within
            else min(rows, key=lambda r: r["false_alarm_rate"]))

    print(f"\n--- confidence floor, chosen on {len(val_eps)} held-out episodes ---")
    print(f"  {'floor':>7s} {'false alarms':>14s} {'macro-F1':>10s}")
    for r in rows:
        mark = " <-- chosen" if r is best else ""
        flag = "" if r["false_alarm_rate"] <= FALSE_ALARM_BUDGET else "  (over budget)"
        print(f"  {r['conf_floor']:>7.2f} {r['false_alarm_rate']:>14.4f} "
              f"{r['macro_f1']:>10.3f}{flag}{mark}")
    if not within:
        print(f"  nothing met the {FALSE_ALARM_BUDGET:.0%} budget; took the "
              f"lowest false-alarm setting instead.")
    return float(best["conf_floor"]), rows


def generate_episodes(n_episodes: int, seed: int):
    """
    Run a mix of controllers/tiers, half clean and half with a pinned
    (fault_type, sensor) fault so every class -- including "none" -- is
    well represented. Returns a list of episode dicts.

    Episode seeds are drawn disjoint from the held-out evaluation seeds
    (config.get_eval_trial_seeds) so this training data can never overlap
    the seeds used to report final A/B/D/E success rates -- no leakage
    between fault-detector training and controller evaluation.
    """
    from config import get_eval_trial_seeds
    from config import sample_disjoint_seeds
    rng = np.random.default_rng(seed)
    eval_seeds = set(get_eval_trial_seeds())
    ep_seeds = sample_disjoint_seeds(rng, n_episodes, exclude=eval_seeds)

    controllers = [BaselineController(), HandTunedFLC()]
    episodes = []
    for ep_idx in range(n_episodes):
        tier = "sparse" if ep_idx % 2 == 0 else "dense"
        controller = controllers[ep_idx % len(controllers)]
        ep_seed = ep_seeds[ep_idx]

        slot = ep_idx % len(EPISODE_PLAN)
        ftype = EPISODE_PLAN[slot]
        if ftype is None:
            # clean episode -> all "none" labels, still valuable negatives
            sim = RobotSimulator(tier, seed=ep_seed, faulty=False)
        else:
            spec = sample_fault_spec_typed(ep_seed, fault_type=ftype)
            sim = RobotSimulator(tier, seed=ep_seed, faulty=True, spec_override=spec)

        result = sim.run(controller)
        episodes.append({
            "episode_id": ep_idx,
            "sensor_log": result.sensor_log,
            "clean_log": result.clean_sensor_log,
            "fault_spec": result.fault_spec,
        })
    return episodes


def main() -> None:
    print(f"Generating {FAULT_DETECTOR_N_EPISODES} episodes for the fault-detection dataset...")
    episodes = generate_episodes(FAULT_DETECTOR_N_EPISODES, seed=4242)

    rng = np.random.default_rng(0)
    ep_ids = np.array([e["episode_id"] for e in episodes])
    shuffled = rng.permutation(ep_ids)
    n_test = max(1, int(round(0.2 * len(shuffled))))
    n_val = max(1, int(round(0.1 * len(shuffled))))
    test_eps = set(shuffled[:n_test].tolist())
    # A separate slice for choosing the confidence floor. Picking it on the
    # test episodes and then reporting on those same episodes would be
    # choosing the threshold that flatters the number being reported.
    val_eps = set(shuffled[n_test:n_test + n_val].tolist())

    X_train, y_train, X_test, y_test = [], [], [], []
    for ep in episodes:
        spec = ep["fault_spec"]
        n_steps = len(ep["sensor_log"])
        mask = _active_mask(spec, n_steps, ep["sensor_log"], ep["clean_log"])
        fault_type = spec.fault_type if spec is not None else None
        fault_idx = spec.sensor_index if spec is not None else None
        X, y = build_windowed_dataset(ep["sensor_log"], fault_type, fault_idx, mask,
                                       window=FAULT_DETECTOR_WINDOW)
        if ep["episode_id"] in test_eps:
            X_test.append(X); y_test.append(y)
        elif ep["episode_id"] in val_eps:
            continue          # kept aside for choosing the confidence floor
        else:
            X_train.append(X); y_train.append(y)

    X_train = np.vstack(X_train); y_train = np.concatenate(y_train)
    X_test = np.vstack(X_test); y_test = np.concatenate(y_test)

    print(f"Train rows: {len(X_train)} (episodes: "
          f"{len(episodes) - len(test_eps) - len(val_eps)}), "
          f"Test rows: {len(X_test)} (episodes: {len(test_eps)}), "
          f"threshold-fitting episodes: {len(val_eps)}")
    print("Train label distribution:", {c: int((y_train == c).sum()) for c in FAULT_DETECTOR_CLASSES})
    print("Test label distribution:", {c: int((y_test == c).sum()) for c in FAULT_DETECTOR_CLASSES})

    # Trim the healthy class down for training only; the test split above is
    # untouched, so nothing here flatters the reported numbers.
    fault_counts = [int((y_train == c).sum()) for c in np.unique(y_train) if c != "none"]
    if fault_counts:
        cap = int(NONE_TRAIN_RATIO * max(fault_counts))
        none_idx = np.flatnonzero(y_train == "none")
        if len(none_idx) > cap:
            keep = np.random.default_rng(0).choice(none_idx, size=cap, replace=False)
            mask = np.ones(len(y_train), dtype=bool)
            mask[none_idx] = False
            mask[keep] = True
            X_train, y_train = X_train[mask], y_train[mask]
            print(f"Subsampled 'none' training rows {len(none_idx)} -> {cap} "
                  f"(cap = {NONE_TRAIN_RATIO:g}x the largest fault class); "
                  f"train rows now {len(X_train)}")

    model = build_classifier(seed=0)
    # There are 40-100x more "no fault" windows than any actual fault, so
    # without reweighting the model just learns to say "fine" every time and
    # still scores 96%. Full inverse-frequency weighting overcorrects the
    # other way and makes it call a fault on almost nothing. Square-rooting
    # it lands in between.
    balanced_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    sample_weight = np.sqrt(balanced_weight)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    chosen_floor, floor_table = select_conf_floor(model, episodes, val_eps,
                                                   FAULT_DETECTOR_WINDOW)

    # Same model, same test episodes, scored two ways: straight off the
    # model, and through the path the controller actually uses. The second
    # one is what tells you how the fault-aware results will turn out.
    windowed = score(y_test, model.predict(X_test),
                     "WINDOWED classifier (raw per-window argmax)")
    y_true_dep, y_pred_dep = evaluate_deployed(model, episodes, test_eps,
                                                FAULT_DETECTOR_WINDOW,
                                                conf_floor=chosen_floor)
    deployed = score(y_true_dep, y_pred_dep,
                     "DEPLOYED detector (push_and_predict, majority-vote smoothed)")

    print("\n--- windowed vs deployed (what smoothing actually buys) ---")
    print(f"  {'metric':<24s} {'windowed':>10s} {'deployed':>10s} {'delta':>9s}")
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        w, d = windowed[key], deployed[key]
        print(f"  {key:<24s} {w:10.3f} {d:10.3f} {d - w:+9.3f}")
    for lbl in sorted(set(windowed["labels"]) | set(deployed["labels"])):
        wp = windowed["classification_report"].get(lbl, {}).get("precision", float("nan"))
        dp = deployed["classification_report"].get(lbl, {}).get("precision", float("nan"))
        print(f"  {'precision[' + lbl + ']':<24s} {wp:10.3f} {dp:10.3f} {dp - wp:+9.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    detector = FaultDetector(model=model, window=FAULT_DETECTOR_WINDOW,
                              conf_floor=chosen_floor)
    detector.save(FAULT_DETECTOR_PATH)

    # The real-path numbers go at the top level so the confusion matrix
    # figure picks them up automatically. The direct-scoring ones are kept
    # underneath for comparison.
    with open(os.path.join(RESULTS_DIR, "fault_detector_report.json"), "w") as f:
        json.dump({
            "evaluation": "deployed (FaultDetector.push_and_predict, majority-vote smoothed)",
            **{k: v for k, v in deployed.items() if k != "n_rows"},
            "n_scored_rows": deployed["n_rows"],
            "windowed_classifier": windowed,
            "n_train_rows": int(len(X_train)),
            "n_test_rows": int(len(X_test)),
            "n_episodes": FAULT_DETECTOR_N_EPISODES,
            "class_weighting": "sqrt-damped balanced (sqrt of inverse-frequency sample weight)",
            "conf_floor": chosen_floor,
            "conf_floor_selection": floor_table,
            "false_alarm_budget": FALSE_ALARM_BUDGET,
            "none_train_ratio": NONE_TRAIN_RATIO,
            "observability_threshold": OBSERVABILITY_THRESHOLD,
        }, f, indent=2)
    print(f"\nSaved model to {FAULT_DETECTOR_PATH}")
    print(f"Saved report to {os.path.join(RESULTS_DIR, 'fault_detector_report.json')}")


if __name__ == "__main__":
    main()
