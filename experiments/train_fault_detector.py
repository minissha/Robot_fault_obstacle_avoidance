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
from controllers.fault_detector import build_windowed_dataset, build_classifier, FaultDetector


def _active_mask(spec, n_steps: int):
    mask = [False] * n_steps
    if spec is None:
        return mask
    for t in range(n_steps):
        if spec.start_step <= t < spec.start_step + spec.duration:
            mask[t] = True
    return mask


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

        if ep_idx % 2 == 0:
            # clean episode -> all "none" labels, still valuable negative examples
            sim = RobotSimulator(tier, seed=ep_seed, faulty=False)
        else:
            fault_ep_idx = ep_idx // 2  # 0,1,2,3,... only over odd (faulty) episodes
            ftype = FAULT_TYPES[fault_ep_idx % len(FAULT_TYPES)]
            spec = sample_fault_spec_typed(ep_seed, fault_type=ftype)
            sim = RobotSimulator(tier, seed=ep_seed, faulty=True, spec_override=spec)

        result = sim.run(controller)
        episodes.append({
            "episode_id": ep_idx,
            "sensor_log": result.sensor_log,
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
    test_eps = set(shuffled[:n_test].tolist())

    X_train, y_train, X_test, y_test = [], [], [], []
    for ep in episodes:
        spec = ep["fault_spec"]
        n_steps = len(ep["sensor_log"])
        mask = _active_mask(spec, n_steps)
        fault_type = spec.fault_type if spec is not None else None
        fault_idx = spec.sensor_index if spec is not None else None
        X, y = build_windowed_dataset(ep["sensor_log"], fault_type, fault_idx, mask,
                                       window=FAULT_DETECTOR_WINDOW)
        if ep["episode_id"] in test_eps:
            X_test.append(X); y_test.append(y)
        else:
            X_train.append(X); y_train.append(y)

    X_train = np.vstack(X_train); y_train = np.concatenate(y_train)
    X_test = np.vstack(X_test); y_test = np.concatenate(y_test)

    print(f"Train rows: {len(X_train)} (episodes: {len(episodes) - len(test_eps)}), "
          f"Test rows: {len(X_test)} (episodes: {len(test_eps)})")
    print("Train label distribution:", {c: int((y_train == c).sum()) for c in FAULT_DETECTOR_CLASSES})
    print("Test label distribution:", {c: int((y_test == c).sum()) for c in FAULT_DETECTOR_CLASSES})

    model = build_classifier(seed=0)
    # Class-imbalance fix: "none" outnumbers every fault class ~100:1
    # (evidence: prior run's train label distribution). Overall accuracy on
    # such data is dominated by the majority class regardless of minority
    # recall -- exactly the "misleadingly high 96%" failure mode. Balanced
    # (inverse-frequency) sample weighting during training is the standard,
    # non-fabricating fix (no synthetic/duplicated rows, no leakage risk).
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    labels_present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    macro_f1 = f1_score(y_test, y_pred, labels=labels_present, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, labels=labels_present, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    cm_row_normalized = cm.astype(float) / np.maximum(1, cm.sum(axis=1, keepdims=True))

    print(f"\nTest accuracy (majority-dominated, reported for reference only): {acc:.3f}")
    print(f"Test BALANCED accuracy (mean per-class recall): {balanced_acc:.3f}")
    print(f"Test macro-F1 (unweighted mean of per-class F1): {macro_f1:.3f}")
    print(f"Classes present in test: {labels_present}")
    print("Per-class precision/recall/F1:")
    for lbl in labels_present:
        r = report[lbl]
        print(f"  {lbl:<12s} precision={r['precision']:.3f} recall={r['recall']:.3f} "
              f"f1={r['f1-score']:.3f} support={int(r['support'])}")
    print("Confusion matrix (rows=true, cols=pred, raw counts):")
    print(labels_present)
    print(cm)
    print("Confusion matrix (row-normalized = per-class recall):")
    print(np.round(cm_row_normalized, 3))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    detector = FaultDetector(model=model, window=FAULT_DETECTOR_WINDOW)
    detector.save(FAULT_DETECTOR_PATH)

    with open(os.path.join(RESULTS_DIR, "fault_detector_report.json"), "w") as f:
        json.dump({
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
            "labels": labels_present,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_row_normalized": cm_row_normalized.tolist(),
            "classification_report": report,
            "n_train_rows": int(len(X_train)),
            "n_test_rows": int(len(X_test)),
            "n_episodes": FAULT_DETECTOR_N_EPISODES,
            "class_weighting": "balanced (inverse frequency, sklearn compute_sample_weight)",
        }, f, indent=2)
    print(f"\nSaved model to {FAULT_DETECTOR_PATH}")
    print(f"Saved report to {os.path.join(RESULTS_DIR, 'fault_detector_report.json')}")


if __name__ == "__main__":
    main()
