"""
goal_bias.py

Shared perception helpers and the goal-blending policy.

Two jobs. First, boil the seven range readings down to a few numbers every
controller works from -- how much room is there straight ahead, and how much
on each shoulder. Every controller gets the same summary so the comparison
is about what they do with it, not about who reads the sensors more
cleverly. Second, mix a goal-directed steering term into whatever the
controller decided, using one fixed policy for all of them, so "does knowing
where the goal is help" is answered under one rule rather than a different
integration per controller.

The blend is the standard reactive-plus-goal summation (Arkin's motor
schemas): weight the goal term by how clear the path ahead is, and hand
control back to obstacle avoidance as things close in.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from simulation_core import (
    MAX_SPEED, MAX_STEER_DEG, MIN_SPEED, ROBOT_RADIUS, SENSOR_ANGLES_DEG,
    SENSOR_RANGE,
)

# Rays within this many degrees of straight ahead count as "the path in
# front of me". At 25 degrees that is the middle three of the seven.
FRONT_ARC_DEG: float = 25.0

# Distance at which the robot should be fully committed to the goal, and the
# distance at which it should have stopped caring about it. Both are in the
# same units as the range readings. REACTION_DISTANCE is roughly how far the
# robot travels while a turn takes effect: a few steps at full speed plus its
# own body. The old code divided by SENSOR_RANGE (100) instead, which is the
# width of the entire world -- the front ray almost never reads anywhere near
# that, so the goal term was permanently turned down to about a third even in
# completely open space.
STOP_DISTANCE: float = 2.0 * ROBOT_RADIUS + 2.0        # 5.0
REACTION_DISTANCE: float = 2.0 * ROBOT_RADIUS + 4.0 * MAX_SPEED  # 15.0
FREE_DISTANCE: float = 35.0

_ANGLES = np.array(SENSOR_ANGLES_DEG, dtype=np.float64)
_ANGLES_RAD = np.radians(_ANGLES)
_COS = np.cos(_ANGLES_RAD)
_SIN = np.abs(np.sin(_ANGLES_RAD))
_LEFT_MASK = _ANGLES > FRONT_ARC_DEG
_RIGHT_MASK = _ANGLES < -FRONT_ARC_DEG

# Half the width of the lane the robot needs to fit through, plus a little
# margin so it does not shave past things.
CORRIDOR_HALF_WIDTH: float = ROBOT_RADIUS + 1.0


def corridor_clearance(readings: np.ndarray) -> float:
    """How far the robot can carry on straight before something is in the way.

    Directional clearance measured dead ahead. A ray angled off to the side
    reporting 40 units is not reporting an obstacle in the robot's path, it
    is looking past it -- so only readings whose blocked cone actually
    covers straight ahead count.

    Taking the plain minimum across the forward rays instead, which is what
    this did at first, made the robot treat every wall it drove alongside as
    an obstruction: it crawled everywhere and rarely committed to the goal.
    """
    return directional_clearance(readings, 0.0)


def arc_clearances(readings: np.ndarray) -> Tuple[float, float, float]:
    """(ahead, left, right).

    `ahead` is the corridor measure above. The two side figures are the
    tightest reading on each shoulder, which is what the controllers compare
    when deciding which way to peel off.
    """
    r = np.asarray(readings, dtype=np.float64)[:len(_ANGLES)]
    return (corridor_clearance(r),
            float(r[_LEFT_MASK].min()),
            float(r[_RIGHT_MASK].min()))


# Candidate headings the controllers consider, relative to the current one.
CANDIDATE_HEADINGS: np.ndarray = np.array(
    [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0], dtype=np.float64)


def corridor_clearance(readings: np.ndarray) -> float:
    """How far the robot can carry on straight before something is in the way.

    Directional clearance measured dead ahead. A ray angled off to the side
    reporting 40 units is not reporting an obstacle in the robot's path, it
    is looking past it -- so only readings whose blocked cone actually
    covers straight ahead count.

    Taking the plain minimum across the forward rays instead, which is what
    this did at first, made the robot treat every wall it drove alongside as
    an obstruction: it crawled everywhere and rarely committed to the goal.
    """
    return directional_clearance(readings, 0.0)


def arc_clearances(readings: np.ndarray) -> Tuple[float, float, float]:
    """(ahead, left, right).

    `ahead` is the corridor measure above. The two side figures are the
    tightest reading on each shoulder, which is what the controllers compare
    when deciding which way to peel off.
    """
    r = np.asarray(readings, dtype=np.float64)[:len(_ANGLES)]
    return (corridor_clearance(r),
            float(r[_LEFT_MASK].min()),
            float(r[_RIGHT_MASK].min()))


# Candidate headings the controllers consider, relative to the current one.
CANDIDATE_HEADINGS: np.ndarray = np.array(
    [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0], dtype=np.float64)


def directional_clearance(readings: np.ndarray, heading_deg: float) -> float:
    """Room available if the robot turned `heading_deg` and then went straight.

    Each reading blocks a cone rather than a single line. Something 5 units
    away sits across roughly 30 degrees of the robot's view; the same thing
    at 50 units blocks under 3. That widening is what lets a fan of separate
    rays say anything about the directions in between them -- without it,
    any heading that happens to fall in a gap between two rays comes back as
    completely clear, which is both wrong and the dangerous way round.

    This is the enlargement angle from Borenstein and Koren's vector field
    histogram, and comparing a few candidate headings this way beats
    comparing one shoulder reading against the other: that cannot tell a
    wall the robot is driving alongside from one it is heading into.
    """
    r = np.asarray(readings, dtype=np.float64)[:len(_ANGLES)]
    safe = np.maximum(r, CORRIDOR_HALF_WIDTH)
    half_cone = np.arcsin(np.clip(CORRIDOR_HALF_WIDTH / safe, 0.0, 1.0))
    offset = np.abs(_ANGLES_RAD - np.radians(heading_deg))
    blocking = offset <= half_cone
    if not blocking.any():
        return float(SENSOR_RANGE)
    return float((r[blocking] * np.cos(offset[blocking])).min())


def clearance_profile(readings: np.ndarray) -> np.ndarray:
    """directional_clearance at each of CANDIDATE_HEADINGS."""
    return np.array([directional_clearance(readings, h) for h in CANDIDATE_HEADINGS])


def open_direction(readings: np.ndarray) -> float:
    """The heading, in degrees, with the most room around it.

    Each ray votes for its own angle with a weight equal to how far it can
    see, so the result leans towards whichever side is genuinely open
    instead of just comparing two shoulder readings. Used as the escape
    direction when something is close ahead.
    """
    r = np.asarray(readings, dtype=np.float64)[:len(_ANGLES)]
    w = np.clip(r, 0.0, None) ** 2      # squared, so a clear ray dominates
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    return float((w * _ANGLES).sum() / total)


def clearance_speed(front_clear: float, turn_deg: float = 0.0) -> float:
    """How fast to travel given the room ahead and how hard we are turning.

    Ramps from MIN_SPEED when an obstacle is at STOP_DISTANCE up to
    MAX_SPEED once the way ahead is clear for FREE_DISTANCE. The turn term
    knocks a bit more off during sharp corrections, which is the whole
    point of having a speed control at all: slow and tight beats fast and
    wide when threading a gap.
    """
    span = FREE_DISTANCE - STOP_DISTANCE
    t = float(np.clip((front_clear - STOP_DISTANCE) / span, 0.0, 1.0))
    speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * t
    turn_factor = 1.0 - 0.45 * float(np.clip(abs(turn_deg) / MAX_STEER_DEG, 0.0, 1.0))
    return float(np.clip(speed * turn_factor, MIN_SPEED, MAX_SPEED))


def goal_weight(front_clear: float) -> float:
    """How much to trust the goal term, from 0 (obstacle on top of us) to 1."""
    span = FREE_DISTANCE - REACTION_DISTANCE
    return float(np.clip((front_clear - REACTION_DISTANCE) / span, 0.0, 1.0))


def blend_goal_steering(obstacle_steer: float, front_clear: float,
                         goal_distance: float, goal_angle_deg: float) -> float:
    """Mix the controller's avoidance steering with a heading to the goal."""
    w = goal_weight(front_clear)
    goal_steer = float(np.clip(goal_angle_deg, -MAX_STEER_DEG, MAX_STEER_DEG))
    return (1.0 - w) * obstacle_steer + w * goal_steer
