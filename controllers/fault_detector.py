"""
fault_detector.py

Blueprint v2, Section 6.2 (Fault Detection Module -- required).

A lightweight scikit-learn MLPClassifier over hand-engineered rolling-window
features (mean, variance, rate-of-change, time-since-last-change) per
sensor channel, predicting which fault (if any) is currently affecting
each of the 3 sensors: {none, dropout, bias, noise_spike, stale}.

This is a per-sensor, per-timestep classifier: at every timestep t (once
at least WINDOW past readings exist) we build one feature vector per
sensor from that sensor's own trailing window, and predict a single
class label. Kept deliberately separate from the ANN imitator
(controllers/ann_imitator.py), which is a *regression* controller and is
untouched by this module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from config import FAULT_DETECTOR_CLASSES, FAULT_DETECTOR_WINDOW

N_FEATURES_PER_SENSOR = 7  # mean, variance, rate-of-change, time-since-last-change,
                            # exact-repeat-fraction, sibling-sensor deviation, max-abs-jump


def window_features(window: np.ndarray, sibling_windows: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    """
    window: 1D array of the last WINDOW raw/observed readings for ONE sensor
    (oldest first). sibling_windows: the same-length windows for the OTHER
    two sensors at the same timesteps, if available.

    Returns [mean, variance, mean_abs_rate_of_change, time_since_last_change,
    exact_repeat_fraction, sibling_deviation].

    The last two features were added to fix a root-caused confusion (see
    FINAL_REPORT.md Sec 8.3/10.2): the original 4 features could not tell
    "dropout" (frozen at ONE fixed value -- simulation_core.py freezes to
    the reading at fault onset) apart from "stale" (a lagged COPY of a
    still-changing signal -- simulation_core.py replays sensor history
    with a delay), because both have low `rate_of_change` on average, but
    stale's diffs are NOT actually zero, just time-shifted. And "bias"
    (a constant offset added to an otherwise-normal, still-varying
    reading) is statistically invisible to any derivative feature, since
    a constant offset does not change diff magnitude at all -- only the
    raw `mean` carries that signal, and `mean` alone is heavily confounded
    with the robot's actual distance to nearby obstacles.
    - exact_repeat_fraction: fraction of consecutive within-window steps
      whose reading is unchanged to within a small epsilon. Dropout's
      frozen value drives this near 1.0; stale's lagged-but-moving copy
      and ordinary clean/noise variation drive it near 0 -- this is the
      feature that separates dropout from stale, which raw
      time_since_last_change (a single trailing run-length) could not.
    - sibling_deviation: this sensor's window mean minus the mean of the
      other two sensors' window means over the same steps. A bias or
      dropout fault on one sensor breaks the rough geometric consistency
      between the three sensors that clean/noise/stale conditions
      preserve; classical robotics cross-sensor consistency checks use
      exactly this kind of signal, and it is not confounded with absolute
      position the way a single sensor's own mean is.
    - max_abs_jump: largest single-step |diff| in the window. Added
      because after the two features above, "stale" (a lagged copy of a
      still-changing signal -- can include one large jump where the
      window straddles the delayed copy's own transition) was being
      confused with "noise_spike" by mean_abs_rate_of_change alone.
      noise_spike's injected magnitude (15-60 units, config.py) produces
      a peak jump well above anything a delayed-but-otherwise-normal
      reading trajectory produces; mean rate-of-change smooths this peak
      away, so it needs its own feature.
    """
    mean = float(np.mean(window))
    var = float(np.var(window))
    diffs = np.diff(window)
    rate_of_change = float(np.mean(np.abs(diffs))) if len(diffs) else 0.0

    eps = 1e-3
    time_since_change = float(len(window))  # default: never changed within window
    for i in range(len(window) - 1, 0, -1):
        if abs(window[i] - window[i - 1]) > eps:
            time_since_change = float(len(window) - 1 - i)
            break

    repeat_fraction = float(np.mean(np.abs(diffs) <= eps)) if len(diffs) else 0.0
    max_abs_jump = float(np.max(np.abs(diffs))) if len(diffs) else 0.0

    if sibling_windows:
        sibling_mean = float(np.mean([np.mean(w) for w in sibling_windows]))
        sibling_deviation = mean - sibling_mean
    else:
        sibling_deviation = 0.0

    return np.array(
        [mean, var, rate_of_change, time_since_change, repeat_fraction,
         sibling_deviation, max_abs_jump],
        dtype=np.float64,
    )


def build_windowed_dataset(sensor_log: Sequence[Tuple[float, float, float]],
                            fault_type: Optional[str],
                            fault_sensor_index: Optional[int],
                            active_mask: Optional[Sequence[bool]] = None,
                            window: int = FAULT_DETECTOR_WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a length-`window` window over one episode's sensor_log (shape
    (T, 3)) and emit one (features, label) pair PER SENSOR PER TIMESTEP
    once a full window is available.

    `active_mask[t]` (if given) marks whether the episode's single fault
    was actually active at step t (faults have a start_step/duration).
    Only the `fault_sensor_index` channel gets a positive label while
    active; the other two sensors on that same step are always labeled
    "none" (they are unaffected), which is what makes this a genuinely
    per-sensor detector rather than a per-episode one.
    """
    sensor_log = np.asarray(sensor_log, dtype=np.float64)
    T = sensor_log.shape[0]
    X_rows: List[np.ndarray] = []
    y_rows: List[str] = []

    if active_mask is None:
        active_mask = [False] * T

    for t in range(window, T):
        windows = [sensor_log[t - window:t, s] for s in range(3)]
        for s in range(3):
            siblings = [windows[k] for k in range(3) if k != s]
            feats = window_features(windows[s], siblings)
            if fault_type is not None and s == fault_sensor_index and active_mask[t]:
                label = fault_type
            else:
                label = "none"
            X_rows.append(feats)
            y_rows.append(label)

    if not X_rows:
        return np.zeros((0, N_FEATURES_PER_SENSOR)), np.zeros((0,), dtype="<U16")
    return np.vstack(X_rows), np.array(y_rows)


class FaultDetector:
    """Wraps a trained MLPClassifier plus a live rolling buffer for online use."""

    def __init__(self, model: Optional[MLPClassifier] = None, window: int = FAULT_DETECTOR_WINDOW,
                 vote_window: int = 5):
        self.model = model
        self.window = window
        self._buffers: List[List[float]] = [[], [], []]
        # Temporal majority-vote smoothing (fixes isolated single-frame
        # classification spikes, e.g. one noisy timestep flipping a label
        # for exactly one frame): keep the last `vote_window` raw
        # per-timestep predictions per sensor and report the plurality
        # label, with confidence = that label's mean confidence over the
        # window. vote_window=5 is a fixed, undertuned smoothing constant
        # (not fit to any result) -- half of FAULT_DETECTOR_WINDOW=12,
        # short enough not to blur genuine short faults (min duration in
        # config.py is 40 steps).
        self.vote_window = vote_window
        self._raw_label_history: List[List[str]] = [[], [], []]
        self._raw_conf_history: List[List[float]] = [[], [], []]

    def reset(self) -> None:
        self._buffers = [[], [], []]
        self._raw_label_history = [[], [], []]
        self._raw_conf_history = [[], [], []]

    def push_and_predict(self, observed_readings: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Feed one timestep of [front, left, right] readings. Returns
        (predicted_labels, confidences) per sensor -- "none"/0.0 for any
        sensor whose buffer isn't yet full.

        Predictions are temporally smoothed via a rolling plurality vote
        over the last `vote_window` raw per-timestep predictions, which
        removes isolated single-frame spikes without needing per-fault
        duration knowledge at inference time.
        """
        raw_labels = ["none"] * 3
        raw_confs = [0.0] * 3
        for s in range(3):
            self._buffers[s].append(float(observed_readings[s]))
            if len(self._buffers[s]) > self.window:
                self._buffers[s].pop(0)
            if len(self._buffers[s]) == self.window and self.model is not None:
                own_w = np.array(self._buffers[s])
                siblings = [np.array(self._buffers[k]) for k in range(3)
                            if k != s and len(self._buffers[k]) == self.window]
                feats = window_features(own_w, siblings).reshape(1, -1)
                proba = self.model.predict_proba(feats)[0]
                idx = int(np.argmax(proba))
                raw_labels[s] = self.model.classes_[idx]
                raw_confs[s] = float(proba[idx])

        labels = ["none"] * 3
        confs = [0.0] * 3
        for s in range(3):
            self._raw_label_history[s].append(raw_labels[s])
            self._raw_conf_history[s].append(raw_confs[s])
            if len(self._raw_label_history[s]) > self.vote_window:
                self._raw_label_history[s].pop(0)
                self._raw_conf_history[s].pop(0)

            votes = self._raw_label_history[s]
            counts: Dict[str, int] = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            winner = max(counts, key=counts.get)
            labels[s] = winner
            matching_confs = [c for lbl, c in zip(votes, self._raw_conf_history[s]) if lbl == winner]
            confs[s] = float(np.mean(matching_confs)) if matching_confs else 0.0
        return labels, confs

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No trained model to save.")
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str, window: int = FAULT_DETECTOR_WINDOW) -> "FaultDetector":
        model = joblib.load(path)
        return cls(model=model, window=window)


def build_classifier(seed: int = 0) -> MLPClassifier:
    # NOTE: early_stopping is intentionally OFF -- some scikit-learn/numpy
    # combinations raise a TypeError inside MLPClassifier's internal
    # np.isnan(y_pred) check when y_pred is a string-label array (observed
    # in this environment). max_iter is capped instead to bound training time.
    return MLPClassifier(
        hidden_layer_sizes=(24, 12),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=False,
        random_state=seed,
    )
