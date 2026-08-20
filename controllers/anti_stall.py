"""
anti_stall.py

Addresses the audited "circular lockout" finding: purely sensor-reactive
controllers (A_baseline, B_handtuned_flc) receive ONLY [front,left,right]
readings each step -- no pose or goal information -- so they can settle
into a deterministic limit cycle (a well-documented failure mode of local
reactive obstacle avoidance; see Koren & Borenstein 1991). Because the
controller interface intentionally carries no pose/goal info (changing
that would be an architecture change, out of scope here), a stall/lockout
can only be detected from the SENSOR SIGNAL ITSELF: if the same
(rounded) [front,left,right] reading recurs periodically, the robot is
very likely retracing the same loop.

CycleBreaker is a small, deterministic, per-episode piece of internal
controller state (reset alongside the controller's own reset()). It does
NOT change the fuzzy rule base, the crisp threshold logic, or the
simulator -- it only monitors the reading stream and, upon detecting a
repeating signature, overrides the NEXT few steering commands with a
strong turn to break the loop.

Determinism is preserved: the breaker's own RNG is re-seeded with a fixed
constant every reset(), so identical (tier, seed, faulty) simulator runs
still produce bit-identical trajectories (see validate_simulator.py /
test_full_trajectory_determinism).
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


class CycleBreaker:
    def __init__(self, window: int = 80, min_gap: int = 6, tolerance: float = 1.5,
                 perturb_steps: int = 6, perturb_deg: float = 35.0,
                 cooldown: int = 40, rng_seed: int = 12345):
        self.window = window
        self.min_gap = min_gap
        self.tolerance = tolerance
        self.perturb_steps = perturb_steps
        self.perturb_deg = perturb_deg
        self.cooldown = cooldown
        self._rng_seed = rng_seed
        self.reset()

    def reset(self) -> None:
        self._history: Deque[Tuple[float, float, float]] = deque(maxlen=self.window)
        self._rng = np.random.default_rng(self._rng_seed)
        self._perturb_remaining = 0
        self._perturb_sign = 1.0
        self._cooldown_remaining = 0

    def _detect_cycle(self, reading: Tuple[float, float, float]) -> bool:
        if len(self._history) < self.window:
            return False
        arr = np.array(self._history)
        cur = np.array(reading)
        # compare against everything at least min_gap steps in the past
        past = arr[: len(arr) - self.min_gap + 1] if self.min_gap <= len(arr) else arr[:0]
        if len(past) == 0:
            return False
        dists = np.linalg.norm(past - cur, axis=1)
        return bool(np.min(dists) < self.tolerance)

    def maybe_override(self, sensor_readings: np.ndarray, controller_steer: float) -> float:
        """
        Call once per step with the current raw/observed reading and the
        wrapped controller's own steering output. Returns the steering
        angle to actually use (unchanged unless a cycle is detected or a
        perturbation is already in progress).
        """
        reading = (float(sensor_readings[0]), float(sensor_readings[1]), float(sensor_readings[2]))

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if self._perturb_remaining > 0:
            self._perturb_remaining -= 1
            self._history.append(reading)
            return self._perturb_sign * self.perturb_deg

        if self._cooldown_remaining == 0 and self._detect_cycle(reading):
            self._perturb_remaining = self.perturb_steps - 1
            self._perturb_sign = float(self._rng.choice([-1.0, 1.0]))
            self._cooldown_remaining = self.cooldown
            self._history.append(reading)
            return self._perturb_sign * self.perturb_deg

        self._history.append(reading)
        return controller_steer

    def observe(self, sensor_readings: np.ndarray) -> None:
        """Record a reading WITHOUT the option to override -- for steps
        where the wrapped controller took an emergency/safety-critical
        action that must not be preempted by an anti-stall perturbation,
        while still keeping the cycle-detection history accurate."""
        reading = (float(sensor_readings[0]), float(sensor_readings[1]), float(sensor_readings[2]))
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        if self._perturb_remaining > 0:
            self._perturb_remaining -= 1
        self._history.append(reading)
