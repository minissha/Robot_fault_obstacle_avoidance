"""
baseline.py

Controller A: crisp reactive obstacle avoidance. No learning, no fuzzy
logic, just thresholds and if/else on the range readings. It is the
reference point the other three have to beat.

It works from the same clearance profile everyone else gets -- how much
room there is along each of seven candidate headings -- and, like the
others, chooses a speed as well as a steering angle. What makes it the
crisp one is that both choices land in discrete buckets: a heading is
either clear enough or it is not, and there are three speeds. Where the
fuzzy controllers slide smoothly between settings, this one snaps.

When it runs out of room it hands over to the shared escape behaviour in
controllers/anti_stall.py, exactly as the other three do.
"""

from __future__ import annotations

import numpy as np

from controllers.anti_stall import TurnCommit, escape_action
from controllers.goal_bias import (
    CANDIDATE_HEADINGS, blend_goal_steering, clearance_profile, clearance_speed,
)
from simulation_core import MAX_SPEED, MAX_STEER_DEG, N_SENSORS

STOP_THRESHOLD: float = 6.0      # below this, stop and turn on the spot
NEAR_THRESHOLD: float = 12.0     # below this, treat a heading as blocked
FAR_THRESHOLD: float = 35.0      # above this, treat it as wide open

# Three speeds, bracketing the same range the fuzzy controllers can reach so
# neither side wins on top speed alone.
# Four settings rather than three: the lowest is a genuine stop, so this
# controller can also rotate on the spot instead of creeping into whatever
# is in front of it. Same capability the fuzzy controllers have, reached
# by a hard threshold instead of a smooth ramp -- which is the whole
# difference being tested.
STOP_SPEED: float = 0.0
CRAWL_SPEED: float = 0.7
CRUISE_SPEED: float = 1.8
FAST_SPEED: float = MAX_SPEED


class BaselineController:
    """Crisp reactive controller: pick a clear heading, pick a speed, go."""

    def __init__(self) -> None:
        self._commit = TurnCommit()

    def reset(self) -> None:
        self._commit.reset()

    def predict(self, observation: np.ndarray):
        """observation: N_SENSORS range readings, then goal_distance and
        goal_angle_deg. Returns (steering_deg, speed). Positive steering
        turns left."""
        obs = np.asarray(observation, dtype=np.float64)
        readings = obs[:N_SENSORS]
        goal_distance, goal_angle_deg = float(obs[N_SENSORS]), float(obs[N_SENSORS + 1])

        profile = clearance_profile(readings)
        ahead = float(profile[len(profile) // 2])

        # Every candidate heading is either acceptable or it is not -- no
        # weighing up of a slightly-tighter but better-aimed gap. Among the
        # acceptable ones take the straightest, since the goal term gets
        # folded in afterwards. If none clear the bar, take the roomiest.
        acceptable = np.flatnonzero(profile >= NEAR_THRESHOLD)
        if not acceptable.size:
            # Boxed in: hand over to the shared escape behaviour.
            return escape_action(profile, CANDIDATE_HEADINGS, ahead, self._commit,
                                  MAX_STEER_DEG, clearance_speed)
        self._commit.release()
        best = acceptable[np.argmin(np.abs(CANDIDATE_HEADINGS[acceptable]))]
        steer = float(CANDIDATE_HEADINGS[best])

        if ahead < STOP_THRESHOLD:
            speed = STOP_SPEED
        elif ahead < NEAR_THRESHOLD:
            speed = CRAWL_SPEED
        elif ahead < FAR_THRESHOLD:
            speed = CRUISE_SPEED
        else:
            speed = FAST_SPEED

        steer = blend_goal_steering(steer, ahead, goal_distance, goal_angle_deg)
        return float(np.clip(steer, -MAX_STEER_DEG, MAX_STEER_DEG)), float(speed)
