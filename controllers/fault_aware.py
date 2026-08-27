"""
fault_aware.py

A wrapper that puts fault detection in front of any of the fault-blind
controllers (A/B/D/E) and patches up readings it believes are wrong.

The response scales with how confident the detector is rather than
switching on at a fixed cutoff. An earlier version used a hard threshold,
which meant two readings at 0.59 and 0.61 confidence got handled completely
differently for no good reason, and the steering jumped around near that
boundary.

Where a suspect reading gets replaced from is the part that changed most.
It used to guess where the reading was heading based on its own recent
history, which is thin evidence -- if a ray has been frozen for twenty
steps, its history is exactly what you cannot trust. With rays 15 degrees
apart, the two either side of it are looking at almost the same piece of
the world, so the average of those is a far better estimate and it does not
depend on the broken ray at all. That only became available once the fan
was widened; with three rays 45 degrees apart there was no such thing as a
neighbouring ray.

Per-ray trust weights are recorded in `trust_log` as the episode runs. The
weighting is continuous rather than a hard fault/no-fault switch, which
makes it the adaptive-sensor-weighting component in its own right, and the
log is what Figure 13 plots.

The other response is to slow down. A controller that is not sure what it
is looking at should buy itself time, and since speed is something the
controllers choose now, it can. Damping the steering as well sounds like
the same idea but measured worse -- see the constants below.
"""

from __future__ import annotations

import numpy as np

from config import FAULT_DETECTOR_WINDOW
from controllers.fault_detector import FaultDetector, neighbour_indices
from simulation_core import MIN_SPEED, N_SENSORS, SENSOR_RANGE, unpack_action

# All three of these were picked by sweeping them on validation seeds
# (disjoint from the evaluation set) and comparing against the same
# controller running fault-blind. Two of the three came out the opposite way
# round from how this was originally written.
#
# Steering scale at confidence 1.0. Turning this down was meant to make the
# robot cautious when it distrusts its sensors, but it consistently lost to
# leaving steering alone -- once a bad reading has been repaired there is
# nothing left to be cautious about, and damping the response just makes it
# slower to get out of the way. 1.0 means no damping.
MIN_STEER_SCALE = 1.0

# Speed scale at confidence 1.0. This one does earn its place: slowing down
# beat not slowing down in every pairing. Buying time is the useful part of
# not trusting your sensors, not steering less.
MIN_SPEED_SCALE = 0.5

# How many steps in a row a ray has to look faulty before we act on it.
# This used to be 5, back when the detector flagged around 8% of clean
# readings and waiting was the only thing stopping it from making the robot
# worse. The detector is now held to a fitted confidence floor and flags
# under 2%, so waiting no longer buys anything -- it just delays the repair.
# Acting immediately won every comparison.
MIN_SUSTAINED_STEPS = 1


class FaultAwareController:
    """
    Wraps a fault-blind controller. Every step:
      1. Feed the observed readings through the FaultDetector, which
         smooths its own predictions over the last few steps.
      2. Ignore a flag unless the same ray has looked faulty for a few
         steps running, so one-off false alarms trigger nothing.
      3. For rays that do look faulty, mix the reading towards what the
         neighbouring rays imply, weighted by confidence.
      4. Ease off the steering and the speed, again by confidence.
    """

    def __init__(self, base_controller, fault_detector: FaultDetector,
                 min_steer_scale: float = MIN_STEER_SCALE,
                 min_speed_scale: float = MIN_SPEED_SCALE,
                 min_sustained: int = MIN_SUSTAINED_STEPS,
                 n_sensors: int = N_SENSORS):
        self.base = base_controller
        self.detector = fault_detector
        self.min_steer_scale = min_steer_scale
        self.min_speed_scale = min_speed_scale
        self.min_sustained = min_sustained
        self.n_sensors = n_sensors
        self._raw_history: list = []
        self._streak = [0] * n_sensors      # consecutive flagged steps, per ray
        # One row per step: how far each ray is being trusted, 1.0 down to 0.0.
        self.trust_log: list = []
        self.label_log: list = []

    def reset(self) -> None:
        self.base.reset()
        self.detector.reset()
        self._raw_history = []
        self._streak = [0] * self.n_sensors
        self.trust_log = []
        self.label_log = []

    def predict(self, observation: np.ndarray):
        """observation: the range readings, then goal_distance and
        goal_angle_deg. Only the range part is checked and corrected --
        the goal terms come from odometry rather than a range sensor, and
        the fault injector never touches them either."""
        readings = np.asarray(observation, dtype=np.float64).copy()
        self._raw_history.append(readings.copy())
        if len(self._raw_history) > FAULT_DETECTOR_WINDOW:
            self._raw_history.pop(0)

        labels, confs = self.detector.push_and_predict(readings[:self.n_sensors])

        max_conf = 0.0
        trust_row = np.ones(self.n_sensors, dtype=np.float64)
        flagged = {s for s in range(self.n_sensors) if labels[s] != "none"}
        for s in range(self.n_sensors):
            fault_conf = confs[s] if labels[s] != "none" else 0.0

            # Keep counting even below the threshold, so a real fault kicks
            # in as soon as it has been going long enough.
            self._streak[s] = self._streak[s] + 1 if fault_conf > 0.0 else 0
            if self._streak[s] < self.min_sustained:
                continue

            max_conf = max(max_conf, fault_conf)
            trust = 1.0 - fault_conf
            trust_row[s] = trust
            readings[s] = trust * readings[s] + (1.0 - trust) * self._estimate(s, flagged)

        self.trust_log.append(trust_row)
        self.label_log.append(list(labels))

        steer, speed = unpack_action(self.base.predict(readings))
        steer *= 1.0 - max_conf * (1.0 - self.min_steer_scale)
        speed *= 1.0 - max_conf * (1.0 - self.min_speed_scale)
        return float(steer), float(max(MIN_SPEED, speed))

    def _estimate(self, sensor_idx: int, flagged: set) -> float:
        """What this ray would probably read if it were working.

        Neighbouring rays first, skipping any that are themselves under
        suspicion. If both neighbours are out, fall back to carrying the
        ray's own recent trend forward -- weaker, but better than nothing.
        """
        current = self._raw_history[-1]
        usable = [k for k in neighbour_indices(sensor_idx, self.n_sensors)
                  if k not in flagged]
        if usable:
            return float(np.clip(np.mean([current[k] for k in usable]),
                                  0.0, SENSOR_RANGE))
        return self._extrapolate(sensor_idx)

    def _extrapolate(self, sensor_idx: int) -> float:
        """Short-horizon linear extrapolation from this ray's own history."""
        hist = [h[sensor_idx] for h in self._raw_history[:-1]]
        if len(hist) < 2:
            return float(self._raw_history[-1][sensor_idx])
        recent = np.array(hist[-5:])
        slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
        return float(np.clip(recent[-1] + slope, 0.0, SENSOR_RANGE))
