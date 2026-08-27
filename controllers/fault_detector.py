"""
fault_detector.py

Blueprint v2, Section 6.2 (Fault Detection Module -- required).

A lightweight scikit-learn MLPClassifier over hand-engineered rolling-window
features (mean, variance, rate-of-change, time-since-last-change) per
sensor channel, predicting which fault (if any) is currently affecting
each ray in the sensor fan: {none, dropout, bias, noise_spike, stale}.

This is a per-sensor, per-timestep classifier: at every timestep t (once
at least WINDOW past readings exist) we build one feature vector per
sensor from that sensor's own trailing window, and predict a single
class label. Kept deliberately separate from the ANN imitator
(controllers/ann_imitator.py), which is a *regression* controller and is
untouched by this module.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from config import FAULT_DETECTOR_CLASSES, FAULT_DETECTOR_WINDOW
from simulation_core import N_SENSORS, SENSOR_RANGE

N_FEATURES_PER_SENSOR = 11  # see window_features

# Don't call something a fault unless the model is at least this sure.
# Without it roughly 8% of perfectly fine readings were being flagged, and
# since the fault-aware controller reacts to every flag, those false alarms
# were costing it. 0.5 cuts them by about a third for very little accuracy.
# Anything higher keeps cutting false alarms but starts missing real faults.
DEFAULT_CONF_FLOOR: float = 0.5


def window_features(window: np.ndarray,
                     neighbour_windows: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    """
    window: the last WINDOW readings for ONE ray, oldest first.
    neighbour_windows: the same window for the rays either side of it.

    Returns nine numbers:
      mean, variance, mean absolute step-to-step change, time since the
      reading last changed, fraction of steps where it did not change at
      all, mean and spread of the gap to its neighbours, largest single
      jump, and fraction of the window sitting at maximum range.

    Why these. The four original ones (mean, variance, rate of change,
    time since change) could not separate the fault types:

    - dropout freezes the reading at one value, stale replays the real
      reading a few steps late. Both look quiet on average, but stale's
      differences are not actually zero, only shifted. `repeat_fraction`
      splits them: dropout drives it to 1.0, stale leaves it near 0.
    - bias adds a constant offset to a reading that otherwise behaves
      normally. No derivative feature can see that, and the raw mean is
      hopelessly confounded with how far the robot happens to be from a
      wall. What does see it is disagreement with the rays either side:
      neighbouring rays look at nearly the same piece of the world, so an
      offset on one of them shows up immediately as a gap that ought not
      to be there. With rays 15 degrees apart this is a strong signal --
      the old version compared against the average of the other two rays
      45 degrees away, which legitimately differ so much that a real bias
      was buried in the disagreement.
    - noise_spike and stale both produce large single jumps, but only
      noise_spike does it repeatedly. `max_abs_jump` catches the size,
      which mean rate of change smooths away.
    - `saturation_fraction` exists because a ray pointing at open space
      reads exactly SENSOR_RANGE for hundreds of steps, which is
      indistinguishable from being frozen unless the detector knows the
      reading is sitting at the top of its scale.
    - stale is the awkward one. It replays the true reading a few steps
      late, so on its own it looks completely healthy -- the values move
      the way they should, just behind. Against its neighbours it shows up
      as an offset, which is also exactly what bias looks like, and being
      the smaller of the two it kept losing the argument and getting
      called "none". What actually separates them is that the offset is a
      delay: slide the ray's history forward a few steps and it lines back
      up with its neighbours, which is not true of a constant offset and
      not true of a frozen reading either. `best_lag` is how far it has to
      slide, and `lag_gain` is how much better the match gets -- near zero
      for bias, clearly positive for stale.
    """
    window = np.asarray(window, dtype=np.float64)
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
    saturation_fraction = float(np.mean(window >= SENSOR_RANGE - 1e-6))

    if neighbour_windows:
        stacked = np.vstack([np.asarray(w, dtype=np.float64) for w in neighbour_windows])
        neighbour_mean = stacked.mean(axis=0)
        gap = window - neighbour_mean
        neighbour_gap_mean = float(np.mean(gap))
        neighbour_gap_std = float(np.std(gap))
        best_lag, lag_gain = _lag_alignment(window, neighbour_mean)
    else:
        neighbour_gap_mean = 0.0
        neighbour_gap_std = 0.0
        best_lag, lag_gain = 0.0, 0.0

    return np.array(
        [mean, var, rate_of_change, time_since_change, repeat_fraction,
         neighbour_gap_mean, neighbour_gap_std, max_abs_jump, saturation_fraction,
         best_lag, lag_gain],
        dtype=np.float64,
    )


# Widest delay the lag search will look for. The stale fault replays
# readings 5 to 14 steps late; anything past this saturates at MAX_LAG,
# which still reads as "clearly delayed" even if the exact figure is
# capped. Needs the window to be at least twice this to leave enough
# overlapping points for the correlation to mean anything.
MAX_LAG: int = 10


def _lag_alignment(window: np.ndarray, neighbour_mean: np.ndarray) -> Tuple[float, float]:
    """How far this ray's history has to slide to line up with its neighbours.

    Returns (best lag, how much the match improved over not sliding at all).
    A delayed reading lines up much better once shifted; a constant offset
    or a frozen value does not line up any better at any shift.
    """
    n = len(window)
    max_lag = min(MAX_LAG, n // 2)
    if max_lag < 1:
        return 0.0, 0.0

    def match(lag: int) -> float:
        a = window[:n - lag] if lag else window
        b = neighbour_mean[lag:] if lag else neighbour_mean
        if len(a) < 3:
            return 0.0
        # Written out as dot products rather than np.corrcoef/np.std. This
        # runs eleven times per ray per step and the numpy call overhead
        # was two thirds of the detector's whole cost; the arithmetic is
        # the same.
        ad = a - a.mean()
        bd = b - b.mean()
        va = float(ad @ ad)
        vb = float(bd @ bd)
        if va < 1e-12 or vb < 1e-12:
            return 0.0
        return float((ad @ bd) / math.sqrt(va * vb))

    scores = [match(lag) for lag in range(max_lag + 1)]
    best = int(np.argmax(scores))
    return float(best), float(scores[best] - scores[0])


def neighbour_indices(sensor_index: int, n_sensors: int = N_SENSORS) -> List[int]:
    """The rays immediately either side of `sensor_index`."""
    return [i for i in (sensor_index - 1, sensor_index + 1) if 0 <= i < n_sensors]


def build_windowed_dataset(sensor_log: Sequence[Tuple[float, ...]],
                            fault_type: Optional[str],
                            fault_sensor_index: Optional[int],
                            active_mask: Optional[Sequence[bool]] = None,
                            window: int = FAULT_DETECTOR_WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over one episode's sensor_log and emit one (features,
    label) pair per ray per timestep, once a full window is available.

    The window for step t includes t itself, which matters: the live
    detector appends the current reading before predicting, so training it
    on a window that stopped at t-1 taught it to answer a slightly
    different question. Noise spikes especially, since those are often one
    big jump and cutting it off loses the whole signal.

    `active_mask[t]` says whether the fault was actually changing what the
    robot saw at step t. Only the faulty ray gets a fault label; the others
    are labelled "none" on the same step, since nothing is wrong with them.
    That is what makes this per-ray rather than per-episode.
    """
    sensor_log = np.asarray(sensor_log, dtype=np.float64)
    T, n_sensors = sensor_log.shape
    X_rows: List[np.ndarray] = []
    y_rows: List[str] = []

    if active_mask is None:
        active_mask = [False] * T

    for t in range(window - 1, T):
        chunk = sensor_log[t - window + 1:t + 1, :]
        for s in range(n_sensors):
            neighbours = [chunk[:, k] for k in neighbour_indices(s, n_sensors)]
            feats = window_features(chunk[:, s], neighbours)
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
                 vote_window: int = 5, conf_floor: float = DEFAULT_CONF_FLOOR,
                 n_sensors: int = N_SENSORS):
        self.model = model
        self.window = window
        self.conf_floor = conf_floor
        self.n_sensors = n_sensors
        self._buffers: List[List[float]] = [[] for _ in range(n_sensors)]
        # Decide based on the last few predictions rather than just the
        # current one, so a single noisy step can't flip the label by itself.
        # Real faults last 40+ steps so this doesn't hide them. Tried 3
        # through 11 and it made almost no difference, so 5 stays.
        self.vote_window = vote_window
        self._raw_label_history: List[List[str]] = [[] for _ in range(n_sensors)]
        self._raw_conf_history: List[List[float]] = [[] for _ in range(n_sensors)]

    def reset(self) -> None:
        self._buffers = [[] for _ in range(self.n_sensors)]
        self._raw_label_history = [[] for _ in range(self.n_sensors)]
        self._raw_conf_history = [[] for _ in range(self.n_sensors)]

    def push_and_predict(self, observed_readings: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Feed one timestep of range readings, one per ray. Returns
        (predicted_labels, confidences) per ray -- "none"/0.0 for any ray
        whose buffer isn't yet full.

        Predictions are temporally smoothed via a rolling plurality vote
        over the last `vote_window` raw per-timestep predictions, which
        removes isolated single-frame spikes without needing per-fault
        duration knowledge at inference time.
        """
        n = self.n_sensors
        raw_labels = ["none"] * n
        raw_confs = [0.0] * n
        for s in range(n):
            self._buffers[s].append(float(observed_readings[s]))
            if len(self._buffers[s]) > self.window:
                self._buffers[s].pop(0)
        ready = all(len(b) == self.window for b in self._buffers)
        if ready and self.model is not None:
            feats = np.vstack([
                window_features(
                    np.array(self._buffers[s]),
                    [np.array(self._buffers[k]) for k in neighbour_indices(s, n)],
                )
                for s in range(n)
            ])
            proba = self.model.predict_proba(feats)
            idx = np.argmax(proba, axis=1)
            raw_labels = [str(self.model.classes_[i]) for i in idx]
            raw_confs = [float(proba[r, i]) for r, i in enumerate(idx)]

        labels = ["none"] * n
        confs = [0.0] * n
        for s in range(n):
            self._raw_label_history[s].append(raw_labels[s])
            self._raw_conf_history[s].append(raw_confs[s])
            if len(self._raw_label_history[s]) > self.vote_window:
                self._raw_label_history[s].pop(0)
                self._raw_conf_history[s].pop(0)

            votes = self._raw_label_history[s]
            counts: Dict[str, int] = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            plurality_label = max(counts, key=counts.get)
            # Needs an actual majority, not just the most votes. Otherwise
            # a fault label could win on 2 out of 5 votes with the other 3
            # split between wrong answers, which is really a coin flip
            # being reported as a detection. Real faults get a clean
            # majority within a step or two anyway.
            if plurality_label != "none" and counts[plurality_label] > len(votes) / 2:
                winner = plurality_label
            else:
                winner = "none"
            matching_confs = [c for lbl, c in zip(votes, self._raw_conf_history[s]) if lbl == winner]
            conf = float(np.mean(matching_confs)) if matching_confs else 0.0

            # Drop low-confidence guesses here rather than in the caller,
            # so anything using the detector gets the cleaner signal.
            if winner != "none" and conf < self.conf_floor:
                winner, conf = "none", 0.0

            labels[s] = winner
            confs[s] = conf
        return labels, confs

    def save(self, path: str) -> None:
        """Save the model together with its confidence floor.

        The floor gets fitted on held-out episodes at training time, so it
        belongs with the model rather than being re-read from a constant.
        Loading it back from source meant retraining could quietly leave a
        threshold behind that was chosen for a different model.
        """
        if self.model is None:
            raise RuntimeError("No trained model to save.")
        joblib.dump({"model": self.model, "conf_floor": self.conf_floor}, path)

    @classmethod
    def load(cls, path: str, window: int = FAULT_DETECTOR_WINDOW,
             conf_floor: Optional[float] = None,
             n_sensors: int = N_SENSORS) -> "FaultDetector":
        """Rebuild a saved detector.

        The confidence floor comes from the saved file, since it was fitted
        alongside the model; pass conf_floor to override it, which is what
        the threshold sweep does. Window length and ray count are structural
        and come from the code.
        """
        blob = joblib.load(path)
        if isinstance(blob, dict):
            model = blob["model"]
            stored_floor = float(blob.get("conf_floor", DEFAULT_CONF_FLOOR))
        else:
            model, stored_floor = blob, DEFAULT_CONF_FLOOR   # older single-object file
        return cls(model=model, window=window,
                    conf_floor=stored_floor if conf_floor is None else conf_floor,
                    n_sensors=n_sensors)


def build_classifier(seed: int = 0) -> MLPClassifier:
    # NOTE: early_stopping is intentionally OFF -- some scikit-learn/numpy
    # combinations raise a TypeError inside MLPClassifier's internal
    # np.isnan(y_pred) check when y_pred is a string-label array (observed
    # in this environment). max_iter is capped instead to bound training time.
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=600,
        early_stopping=False,
        random_state=seed,
    )
