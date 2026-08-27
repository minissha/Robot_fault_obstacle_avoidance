"""
fuzzy_handtuned.py

Controller B: hand-tuned Mamdani fuzzy logic controller.

Inputs are the same three numbers the crisp baseline works from, read off
the clearance profile: room straight ahead (A), room to the left (L), room
to the right (R), each in [0, 100] and each carrying the terms
{Near, Medium, Far}.

Outputs are a steering angle in [-45, 45] and a speed. The steering side
uses five terms {SharpLeft, Left, Straight, Right, SharpRight}; the speed
side uses three {Slow, Medium, Fast}. Both are defuzzified by centroid, so
unlike the baseline the controller can settle between two settings rather
than having to commit to one -- easing off to two-thirds speed while
drifting ten degrees left is a thing it can express and the crisp version
cannot.

Membership functions are trapezoidal/triangular, defined by two boundary
values per input:
    Near:   1.0 below near_bound, falling to 0 by the midpoint
    Medium: triangular, peak at the midpoint
    Far:    rises from the midpoint, 1.0 at/above far_bound

The eight boundary values
[a_near, a_far, l_near, l_far, r_near, r_far, v_near, v_far]
form the chromosome the NSGA-II variant evolves (controllers/fuzzy_nsga2.py),
which reuses this class with evolved numbers instead of hand-tuned ones.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from simulation_core import (
    MAX_SPEED, MAX_STEER_DEG, MIN_SPEED, N_SENSORS, ROBOT_RADIUS,
)
from controllers.anti_stall import TurnCommit, escape_action
from controllers.goal_bias import (
    CANDIDATE_HEADINGS, blend_goal_steering, clearance_profile, clearance_speed,
)

# Crisp safety reflex, sitting underneath the fuzzy layer. Centroid
# defuzzification is a weighted average over every active rule, so close to
# a boundary in the input space it can return a gentler turn than any single
# active rule actually asks for -- which is exactly the wrong moment for it.
# Derived from kinematics rather than tuned: body diameter plus the distance
# covered while a turn takes effect.
CRITICAL_AHEAD: float = 2.0 * ROBOT_RADIUS + 2.0 * MAX_SPEED   # 9.0

# [a_near, a_far, l_near, l_far, r_near, r_far, v_near, v_far]
DEFAULT_PARAMS: Tuple[float, ...] = (14.0, 38.0, 12.0, 34.0, 12.0, 34.0, 10.0, 32.0)

OUTPUT_MIN: float = -45.0
OUTPUT_MAX: float = 45.0
OUTPUT_RESOLUTION: int = 181  # 0.5-degree steps

OUTPUT_CENTERS: Dict[str, float] = {
    "SharpLeft": 40.0,
    "Left": 20.0,
    "Straight": 0.0,
    "Right": -20.0,
    "SharpRight": -40.0,
}
OUTPUT_SPREAD: float = 22.0  # triangular half-width for the steering terms

SPEED_CENTERS: Dict[str, float] = {
    "Slow": MIN_SPEED,
    "Medium": 0.5 * (MIN_SPEED + MAX_SPEED),
    "Fast": MAX_SPEED,
}
SPEED_SPREAD: float = 0.5 * (MAX_SPEED - MIN_SPEED)
SPEED_RESOLUTION: int = 121


def _trap_near(x: float, near_bound: float, far_bound: float) -> float:
    """Near: 1.0 at/below near_bound, falling linearly to 0 by the midpoint."""
    mid = (near_bound + far_bound) / 2.0
    if x <= near_bound:
        return 1.0
    if x >= mid:
        return 0.0
    return (mid - x) / (mid - near_bound)


def _tri_medium(x: float, near_bound: float, far_bound: float) -> float:
    """Medium: triangular, peaking at the midpoint between the bounds."""
    mid = (near_bound + far_bound) / 2.0
    half_width = (far_bound - near_bound) / 2.0
    if half_width <= 1e-9:
        return 1.0 if x == mid else 0.0
    dist = abs(x - mid)
    if dist >= half_width:
        return 0.0
    return 1.0 - dist / half_width


def _trap_far(x: float, near_bound: float, far_bound: float) -> float:
    """Far: 0 below the midpoint, rising to 1.0 at/above far_bound."""
    mid = (near_bound + far_bound) / 2.0
    if x >= far_bound:
        return 1.0
    if x <= mid:
        return 0.0
    return (x - mid) / (far_bound - mid)


def fuzzify(x: float, near_bound: float, far_bound: float) -> Dict[str, float]:
    """Membership degrees for {Near, Medium, Far} for one input value."""
    return {
        "Near": _trap_near(x, near_bound, far_bound),
        "Medium": _tri_medium(x, near_bound, far_bound),
        "Far": _trap_far(x, near_bound, far_bound),
    }


# --------------------------------------------------------------------------
# Rule base: all 27 combinations of {Near, Medium, Far} over (A, L, R).
#
# The original rule base only listed 15 of the 27. For any combination it
# missed, inference produced an all-zero output and defuzzification quietly
# returned 0.0 -- "Straight" -- so the controller drove into things whenever
# the readings landed in an uncovered corner of the input space. Replaying
# failed episodes confirmed it: they collided while an uncovered combination
# held for several steps and the controller kept commanding Straight into a
# closing obstacle. Every combination is covered now, under one consistent
# policy: turn towards whichever side has more room, and turn harder the
# tighter it gets ahead.
# --------------------------------------------------------------------------

_TERMS: Tuple[str, str, str] = ("Near", "Medium", "Far")
_ORDER: Dict[str, int] = {"Near": 0, "Medium": 1, "Far": 2}


def _steer_policy(a_term: str, l_term: str, r_term: str) -> str:
    """Which way to turn, for one combination of input terms."""
    if a_term == "Near":
        if _ORDER[l_term] == _ORDER[r_term]:
            return "SharpLeft"          # symmetric tie-break (dead end)
        return "SharpLeft" if _ORDER[l_term] > _ORDER[r_term] else "SharpRight"
    if a_term == "Medium":
        if _ORDER[l_term] > _ORDER[r_term]:
            return "Left"
        if _ORDER[r_term] > _ORDER[l_term]:
            return "Right"
        return "Straight"
    # Ahead is clear: only react to a flank that is closing in.
    if l_term == "Near" and r_term != "Near":
        return "Right"
    if r_term == "Near" and l_term != "Near":
        return "Left"
    if l_term == "Near" and r_term == "Near":
        return "Right"
    return "Straight"


def _speed_policy(a_term: str, l_term: str, r_term: str) -> str:
    """How fast to travel, for one combination of input terms."""
    if a_term == "Near":
        return "Slow"
    if a_term == "Medium":
        return "Slow" if "Near" in (l_term, r_term) else "Medium"
    if "Near" in (l_term, r_term):
        return "Medium"
    return "Fast"


RULES: List[Tuple[str, str, str, str, str]] = [
    (a, l, r, _steer_policy(a, l, r), _speed_policy(a, l, r))
    for a in _TERMS for l in _TERMS for r in _TERMS
]
assert len(RULES) == 27, "rule base must cover all 27 (A, L, R) term combinations"

_LEFT_CANDIDATES = CANDIDATE_HEADINGS > 0
_RIGHT_CANDIDATES = CANDIDATE_HEADINGS < 0
_AHEAD_CANDIDATE = int(np.argmin(np.abs(CANDIDATE_HEADINGS)))


class HandTunedFLC:
    """Mamdani fuzzy controller over the clearance profile."""

    def __init__(self, params: Tuple[float, ...] = DEFAULT_PARAMS):
        if len(params) != 8:
            raise ValueError(
                "FLC params must have exactly 8 values: [a_near, a_far, "
                "l_near, l_far, r_near, r_far, v_near, v_far].")
        self.params = tuple(float(p) for p in params)
        self._steer_universe = np.linspace(OUTPUT_MIN, OUTPUT_MAX, OUTPUT_RESOLUTION)
        self._speed_universe = np.linspace(MIN_SPEED, MAX_SPEED, SPEED_RESOLUTION)
        self._steer_mf = {t: np.clip(1.0 - np.abs(self._steer_universe - c) / OUTPUT_SPREAD, 0.0, 1.0)
                           for t, c in OUTPUT_CENTERS.items()}
        self._speed_mf = {t: np.clip(1.0 - np.abs(self._speed_universe - c) / SPEED_SPREAD, 0.0, 1.0)
                           for t, c in SPEED_CENTERS.items()}
        self._commit = TurnCommit()

    def reset(self) -> None:
        self._commit.reset()

    @staticmethod
    def _inputs(readings: np.ndarray) -> Tuple[float, float, float]:
        """(ahead, left, right) read off the clearance profile.

        The sides take the best heading available on that side rather than
        the worst, because the question being asked is "is there a way
        through over there", not "is anything over there at all".
        """
        profile = clearance_profile(readings)
        return (float(profile[_AHEAD_CANDIDATE]),
                float(profile[_LEFT_CANDIDATES].max()),
                float(profile[_RIGHT_CANDIDATES].max()))

    def predict(self, observation: np.ndarray):
        """observation: N_SENSORS range readings, then goal_distance and
        goal_angle_deg. Returns (steering_deg, speed)."""
        obs = np.asarray(observation, dtype=np.float64)
        readings = obs[:N_SENSORS]
        goal_distance, goal_angle_deg = float(obs[N_SENSORS]), float(obs[N_SENSORS + 1])

        ahead, left, right = self._inputs(readings)

        if ahead < CRITICAL_AHEAD:
            # Reflex: never diluted by the goal term, never pre-empted by
            # the cycle-breaker. Hardest turn towards whichever side has
            # room, and slow enough to make it.
            #
            # Speed comes off the same ramp the fuzzy layer uses rather than
            # being pinned at the floor. That matters now the floor is zero:
            # pinned, the robot would stop dead and stay stopped: on the
            # ramp it stops while it is facing the obstacle and picks up
            # again the moment the turn has opened the way ahead.
            return escape_action(clearance_profile(readings), CANDIDATE_HEADINGS,
                                  ahead, self._commit, MAX_STEER_DEG, clearance_speed)

        a_near, a_far, l_near, l_far, r_near, r_far, v_near, v_far = self.params

        self._commit.release()      # way ahead is clear; next jam decides afresh
        a_mf = fuzzify(ahead, a_near, a_far)
        l_mf = fuzzify(left, l_near, l_far)
        r_mf = fuzzify(right, r_near, r_far)
        v_mf = fuzzify(ahead, v_near, v_far)

        steer_agg = np.zeros_like(self._steer_universe)
        speed_agg = np.zeros_like(self._speed_universe)

        for a_term, l_term, r_term, steer_term, speed_term in RULES:
            strength = min(a_mf[a_term], l_mf[l_term], r_mf[r_term])
            if strength <= 0.0:
                continue
            np.maximum(steer_agg, np.minimum(self._steer_mf[steer_term], strength),
                        out=steer_agg)
            # The speed side reads "room ahead" through its own boundaries, so
            # a rule can fire hard for steering and only softly for speed.
            speed_strength = min(strength, v_mf[a_term])
            np.maximum(speed_agg, np.minimum(self._speed_mf[speed_term], speed_strength),
                        out=speed_agg)

        steer_total = float(steer_agg.sum())
        centroid = 0.0 if steer_total <= 1e-9 else float(
            (steer_agg * self._steer_universe).sum() / steer_total)

        speed_total = float(speed_agg.sum())
        speed = MIN_SPEED if speed_total <= 1e-9 else float(
            (speed_agg * self._speed_universe).sum() / speed_total)

        steer = blend_goal_steering(centroid, ahead, goal_distance, goal_angle_deg)
        return (float(np.clip(steer, -MAX_STEER_DEG, MAX_STEER_DEG)),
                float(np.clip(speed, MIN_SPEED, MAX_SPEED)))
