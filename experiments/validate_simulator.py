"""
validate_simulator.py

Internal-validity checks for the simulation environment.

Run from the project root:
    python experiments/validate_simulator.py

These tests deliberately use analytically known layouts rather than random
obstacle fields. They are intended to answer a reviewer question:

    "How do you know the simulator itself behaves correctly?"

Checks:
1. Empty world: the baseline controller reaches the goal with no obstacles.
2. Single known obstacle: analytically determine whether the straight
   start-goal segment intersects the obstacle, then verify the corresponding
   sensor geometry and controller outcome on a known passable layout.
3. Straight-line path: in an empty world, path_length + final_goal_distance
   must equal the analytical start-goal distance (within numerical tolerance).

The tests avoid changing the production simulator. A tiny fixed-layout
subclass is used only to replace the generated obstacle list.
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

import numpy as np

# Allow execution as:
#     python experiments/validate_simulator.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_core import (
    GOAL_POS,
    ROBOT_RADIUS,
    START_POS,
    WORLD_SIZE,
    FaultInjector,
    Obstacle,
    RobotSimulator,
    cast_ray,
    get_sensor_readings,
)
from controllers.baseline import BaselineController
from controllers.fuzzy_handtuned import HandTunedFLC


TOL = 1e-6


class StraightController:
    """Deterministic zero-steering controller for analytical path checks."""

    def reset(self) -> None:
        return None

    def predict(self, sensor_readings: np.ndarray) -> float:
        return 0.0


class FixedWorldSimulator(RobotSimulator):
    """RobotSimulator with a caller-specified obstacle list."""

    def __init__(self, obstacles: List[Obstacle], seed: int = 0):
        self.tier = "fixed"
        self.seed = seed
        self.faulty = False
        self.wide_sensors = False  # this subclass bypasses RobotSimulator.__init__
        # entirely, so it must set every attribute RobotSimulator.run() reads,
        # not just the ones that existed before wide_sensors was added -- an
        # attribute added to the base class is silently missing from any
        # subclass that skips super().__init__(), which is exactly what
        # broke here (AttributeError caught by validate_simulator.py's own
        # test run this session).
        self.obstacles = list(obstacles)
        self.fault_injector = FaultInjector(faulty=False, seed=seed)


def _orientation(a: Tuple[float, float],
                 b: Tuple[float, float],
                 c: Tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segment_intersects_rect(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    rect: Obstacle,
) -> bool:
    """Exact segment-vs-axis-aligned-rectangle test using slab clipping."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    t0, t1 = 0.0, 1.0

    for p, d, lo, hi in (
        (p0[0], dx, rect.x_min, rect.x_max),
        (p0[1], dy, rect.y_min, rect.y_max),
    ):
        if abs(d) < TOL:
            if p < lo or p > hi:
                return False
            continue

        a = (lo - p) / d
        b = (hi - p) / d
        if a > b:
            a, b = b, a

        t0 = max(t0, a)
        t1 = min(t1, b)
        if t0 > t1:
            return False

    return True


def distance_point_to_segment(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Euclidean distance from p to line segment a-b."""
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= TOL:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / denom
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy)


def test_empty_world_success() -> None:
    """
    Known analytical case: with no obstacles, the initial heading is exactly
    toward the goal. The baseline controller should command zero steering and
    reach the goal every time.

    ALSO checks the hand-tuned FLC here, not just the baseline: this exact
    check (added post-hoc) is what caught a real bug during development --
    an earlier FLC parameterization treated the world-boundary wall near the
    goal corner as a phantom obstacle and steered away before reaching the
    goal, scoring 0/N in an OBSTACLE-FREE world. Checking every controller
    against this analytical case, not just the baseline, is the point.
    """
    expected_distance = math.dist(START_POS, GOAL_POS)

    for name, controller_cls in (("baseline", BaselineController), ("hand_tuned_flc", HandTunedFLC)):
        n_success = 0
        n_seeds = 20
        for seed in range(n_seeds):
            result = FixedWorldSimulator([], seed=seed).run(controller_cls())
            if result.metrics.success and not result.metrics.collision:
                n_success += 1
        assert n_success >= int(0.9 * n_seeds), (
            f"{name} should reach the goal in an obstacle-free world almost "
            f"every time, got {n_success}/{n_seeds}"
        )

    print("[PASS] empty world: baseline and hand-tuned FLC both reach the goal "
          "(>=90% of seeds) with zero obstacles")


def test_single_known_obstacle_geometry_and_controller() -> None:
    """
    Known passable layout.

    The obstacle is centered at (77.5, 17.5), deliberately offset from the
    start-goal diagonal y=x. The analytical start-goal segment therefore does
    not intersect it. Its minimum distance from the direct path is also much
    larger than the robot radius, so the obstacle cannot geometrically block
    the ideal trajectory.

    We then verify:
      1) the analytical geometry predicts "direct path clear";
      2) the front sensor from the start points toward the goal and therefore
         does not report the offset obstacle;
      3) the baseline controller succeeds without collision.

    This is a stronger internal-validity check than merely observing that one
    random seed happened to succeed.
    """
    obstacle = Obstacle(75.0, 15.0, 80.0, 20.0)
    direct_path = not segment_intersects_rect(START_POS, GOAL_POS, obstacle)

    assert direct_path, "known offset obstacle should not intersect start-goal path"

    # Check clearance of the obstacle center from the ideal path.
    center = (
        (obstacle.x_min + obstacle.x_max) / 2.0,
        (obstacle.y_min + obstacle.y_max) / 2.0,
    )
    path_clearance = distance_point_to_segment(center, START_POS, GOAL_POS)
    assert path_clearance > ROBOT_RADIUS, (
        f"obstacle too close to ideal path: clearance={path_clearance:.3f}"
    )

    raw = get_sensor_readings(
        START_POS[0],
        START_POS[1],
        math.atan2(
            GOAL_POS[1] - START_POS[1],
            GOAL_POS[0] - START_POS[0],
        ),
        [obstacle],
    )

    # The obstacle is deliberately outside the front ray; the front reading
    # should therefore be the world-boundary distance, not obstacle distance.
    # The world-boundary intersection lies beyond SENSOR_RANGE, so the
    # documented ray-caster cap of 100.0 applies.
    expected_front_reading = 100.0
    assert abs(raw[0] - expected_front_reading) < 1e-9, (
        f"unexpected front-ray distance: {raw[0]:.6f} "
        f"(expected {expected_front_reading:.6f})"
    )

    for seed in range(10):
        result = FixedWorldSimulator([obstacle], seed=seed).run(BaselineController())
        assert result.metrics.success, (
            f"baseline should succeed on the analytically passable known layout "
            f"(seed={seed})"
        )
        assert not result.metrics.collision

    print(
        "[PASS] single known obstacle: analytical path-clear prediction, "
        "sensor geometry, and controller success agree"
    )


def test_straight_line_path_length() -> None:
    """
    In an empty world the robot travels in a straight line at constant speed.
    Because the simulator stops inside GOAL_RADIUS, the exact identity is:

        path_length + final_goal_distance == start_goal_distance

    up to floating-point integration error.
    """
    expected_distance = math.dist(START_POS, GOAL_POS)
    result = FixedWorldSimulator([]).run(StraightController())

    reconstructed_distance = (
        result.metrics.path_length + result.metrics.final_goal_distance
    )

    error = abs(reconstructed_distance - expected_distance)
    assert error < 1e-9, (
        f"straight-line distance mismatch: expected {expected_distance:.12f}, "
        f"got {reconstructed_distance:.12f}, error={error:.3e}"
    )

    # Also ensure the actual path is collinear with start and goal.
    max_deviation = 0.0
    for point in result.trajectory:
        max_deviation = max(
            max_deviation,
            abs(_orientation(START_POS, GOAL_POS, point))
            / expected_distance,
        )

    assert max_deviation < 1e-9, (
        f"trajectory deviates from straight line: {max_deviation:.3e}"
    )

    print(
        "[PASS] straight-line path: analytical distance matches simulation "
        f"(error={error:.2e})"
    )


def test_world_bounds_sanity() -> None:
    """Basic check that the fixed validation world uses the documented bounds."""
    assert WORLD_SIZE == 100.0
    assert START_POS == (8.0, 8.0)
    assert GOAL_POS == (92.0, 92.0)
    print("[PASS] documented world/start/goal constants are consistent")


def main() -> None:
    print("Simulator internal-validity validation")
    print("=" * 45)
    test_empty_world_success()
    test_single_known_obstacle_geometry_and_controller()
    test_straight_line_path_length()
    test_world_bounds_sanity()
    print("\nALL SIMULATOR VALIDATION TESTS PASSED")


if __name__ == "__main__":
    main()
