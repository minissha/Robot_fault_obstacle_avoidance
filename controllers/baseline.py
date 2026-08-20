"""
baseline.py

Controller A: crisp reactive threshold-based obstacle avoidance.
No learning, no fuzzy logic — pure if/else logic on raw sensor distances.

Audit fix (circular lockouts): wrapped with CycleBreaker (see
controllers/anti_stall.py) -- detects a repeating sensor-reading
signature (the only lockout signal available to a sensor-only
controller) and injects a short strong-turn perturbation to break the
loop. Core threshold logic below is UNCHANGED.
"""

from __future__ import annotations

import numpy as np

from controllers.anti_stall import CycleBreaker

NEAR_THRESHOLD: float = 25.0
FAR_THRESHOLD: float = 55.0
SHARP_TURN_DEG: float = 40.0
GENTLE_TURN_DEG: float = 15.0


class BaselineController:
    """Crisp reactive controller: if a sensor reads 'near', turn away."""

    def __init__(self) -> None:
        self._breaker = CycleBreaker(rng_seed=12345)

    def reset(self) -> None:
        self._breaker.reset()

    def predict(self, sensor_readings: np.ndarray) -> float:
        """
        Map [front, left, right] readings to a steering angle in degrees.
        Positive angle = turn left, negative = turn right.
        """
        front, left, right = float(sensor_readings[0]), float(sensor_readings[1]), float(sensor_readings[2])

        if front < NEAR_THRESHOLD:
            # Obstacle directly ahead: turn toward the side with more clearance.
            steer = SHARP_TURN_DEG if left > right else -SHARP_TURN_DEG
        elif left < NEAR_THRESHOLD and right >= NEAR_THRESHOLD:
            steer = -GENTLE_TURN_DEG
        elif right < NEAR_THRESHOLD and left >= NEAR_THRESHOLD:
            steer = GENTLE_TURN_DEG
        elif left < NEAR_THRESHOLD and right < NEAR_THRESHOLD:
            steer = SHARP_TURN_DEG if left > right else -SHARP_TURN_DEG
        elif front > FAR_THRESHOLD and left > FAR_THRESHOLD and right > FAR_THRESHOLD:
            steer = 0.0
        else:
            # Mild bias toward the more open side to keep progressing toward goal.
            steer = (left - right) * 0.15

        return self._breaker.maybe_override(sensor_readings, steer)
