"""
fault_aware.py

Blueprint v2, Section 6.2: fault-aware wrapper around any existing
fault-blind controller (A/B/D/E).

REVISED (this pass): the earlier version used a hard binary threshold
(`confidence >= FAULT_DETECTOR_CONF_THRESHOLD` -> fully substitute the
reading and apply a fixed conservative-steering multiplier). That produces
a step discontinuity right at the threshold -- two frames with confidence
0.59 and 0.61 get treated completely differently -- which is exactly the
kind of instability that shows up as erratic control near the boundary.

Replaced with CONTINUOUS confidence-proportional trust weighting, per
sensor, every step:

    w_i = 1 - confidence_i(fault)          (trust weight in [0, 1])
    corrected_reading_i = w_i * raw_i + (1 - w_i) * extrapolated_i

and a continuous conservative-steering scale (instead of a fixed 0.5x
cliff-edge):

    steer_scale = 1 - max_i(confidence_i(fault)) * (1 - MIN_STEER_SCALE)

No behavior changes discontinuously at any confidence value; a
low-confidence flag barely perturbs the reading/steering, a
high-confidence flag approaches (but at max confidence=1.0, still doesn't
fully exceed) full substitution/full conservatism.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import FAULT_DETECTOR_WINDOW
from controllers.fault_detector import FaultDetector

MIN_STEER_SCALE = 0.4  # steering scale at confidence=1.0 (never fully zeroed out)


class FaultAwareController:
    """
    Wraps a fault-blind controller. Every step:
      1. Feed the raw observed reading through the FaultDetector (which
         internally applies temporal majority-vote smoothing -- see
         controllers/fault_detector.py).
      2. For each sensor, compute a continuous trust weight
         w_i = 1 - confidence_i and blend the raw reading with a
         short-horizon linear extrapolation using that weight (fallback
         substitution, option (a) from the blueprint, made continuous).
      3. Scale the wrapped controller's steering output continuously by
         `1 - max_confidence * (1 - MIN_STEER_SCALE)` (confidence-gated
         behavior, option (b), made continuous instead of a hard cliff).
    """

    def __init__(self, base_controller, fault_detector: FaultDetector,
                 min_steer_scale: float = MIN_STEER_SCALE):
        self.base = base_controller
        self.detector = fault_detector
        self.min_steer_scale = min_steer_scale
        self._raw_history: list = []

    def reset(self) -> None:
        self.base.reset()
        self.detector.reset()
        self._raw_history = []

    def predict(self, sensor_readings: np.ndarray) -> float:
        readings = np.asarray(sensor_readings, dtype=np.float64).copy()
        self._raw_history.append(readings.copy())
        if len(self._raw_history) > FAULT_DETECTOR_WINDOW:
            self._raw_history.pop(0)

        labels, confs = self.detector.push_and_predict(readings)

        max_conf = 0.0
        for s in range(3):
            fault_conf = confs[s] if labels[s] != "none" else 0.0
            max_conf = max(max_conf, fault_conf)
            if fault_conf > 0.0:
                trust = 1.0 - fault_conf  # in [0, 1]
                extrapolated = self._extrapolate(s)
                readings[s] = trust * readings[s] + (1.0 - trust) * extrapolated

        steer_scale = 1.0 - max_conf * (1.0 - self.min_steer_scale)
        steer = float(self.base.predict(readings)) * steer_scale
        return steer

    def _extrapolate(self, sensor_idx: int) -> float:
        """Short-horizon linear extrapolation from recent (pre-window) history."""
        hist = [h[sensor_idx] for h in self._raw_history[:-1]]
        if len(hist) < 2:
            return float(self._raw_history[-1][sensor_idx])
        recent = np.array(hist[-5:])
        if len(recent) >= 2:
            slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
            return float(np.clip(recent[-1] + slope, 0.0, 100.0))
        return float(recent[-1])
