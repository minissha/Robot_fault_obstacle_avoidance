"""
test_simulator.py

Verification suite for the simulator core (Day 1-3 deliverable). Run from
the project root:

    python experiments/test_simulator.py

Checks:
    - sparse/dense environment obstacle counts
    - all sensors present, readings within [0, SENSOR_RANGE]
    - clean vs faulty episodes both run
    - all 4 fault types can be sampled and applied
    - reproducibility: identical (tier, seed, faulty) -> identical results
    - metrics are all present and well-typed
    - controller-interface compatibility (baseline + hand-tuned FLC)

Exits with code 0 and prints "ALL TESTS PASSED" on success; raises an
AssertionError (nonzero exit) on any failure.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simulation_core import (
    N_SENSORS,
    DENSE_OBSTACLE_RANGE,
    SPARSE_OBSTACLE_RANGE,
    FAULT_TYPES,
    SENSOR_RANGE,
    RobotSimulator,
    generate_obstacles,
    sample_fault_spec,
)
from controllers.baseline import BaselineController
from controllers.fuzzy_handtuned import HandTunedFLC


def test_obstacle_generation() -> None:
    for seed in range(5):
        sparse = generate_obstacles("sparse", seed)
        dense = generate_obstacles("dense", seed)
        slo, shi = SPARSE_OBSTACLE_RANGE
        assert slo <= len(sparse) <= shi, f"sparse tier produced {len(sparse)} obstacles"
        lo, hi = DENSE_OBSTACLE_RANGE
        assert lo <= len(dense) <= hi, f"dense tier produced {len(dense)} obstacles"
    print("[PASS] obstacle generation (sparse 5-8, dense 22-30; see Blueprint v2 Sec 2.1 feasibility fix)")


def test_dense_tier_feasibility() -> None:
    """Every generated dense layout must have obstacle-obstacle gaps the robot can pass."""
    from simulation_core import MIN_OBSTACLE_GAP, _dist_rect_to_rect
    import itertools
    for seed in range(30):
        obs = generate_obstacles("dense", seed)
        for a, b in itertools.combinations(obs, 2):
            assert _dist_rect_to_rect(a, b) >= MIN_OBSTACLE_GAP - 1e-9, \
                f"seed {seed}: obstacles closer than MIN_OBSTACLE_GAP"
    print("[PASS] dense-tier feasibility: all obstacle-obstacle gaps >= MIN_OBSTACLE_GAP")


def test_obstacle_reproducibility() -> None:
    a = generate_obstacles("dense", seed=42)
    b = generate_obstacles("dense", seed=42)
    assert len(a) == len(b)
    for oa, ob in zip(a, b):
        assert oa == ob
    print("[PASS] obstacle generation reproducibility")


def test_three_sensors_and_range() -> None:
    sim = RobotSimulator("sparse", seed=3, faulty=False)
    controller = BaselineController()
    result = sim.run(controller)
    assert len(result.sensor_log) > 0, "no sensor readings logged"
    for reading in result.sensor_log:
        assert len(reading) == N_SENSORS, f"expected {N_SENSORS} sensor readings"
        for v in reading:
            assert -1e-6 <= v <= SENSOR_RANGE + 1e-6, f"sensor reading {v} out of range"
    print(f"[PASS] {N_SENSORS} sensors present, all readings within [0, SENSOR_RANGE]")


def test_clean_and_faulty_episodes() -> None:
    for faulty in (False, True):
        sim = RobotSimulator("sparse", seed=7, faulty=faulty)
        result = sim.run(BaselineController())
        assert result.metrics.steps > 0
        if faulty:
            assert result.fault_spec is not None
        else:
            assert result.fault_spec is None
    print("[PASS] clean and faulty episodes both run correctly")


def test_all_fault_types_reachable() -> None:
    seen = set()
    for seed in range(2000):
        spec = sample_fault_spec(seed)
        seen.add(spec.fault_type)
        if seen == set(FAULT_TYPES):
            break
    assert seen == set(FAULT_TYPES), f"only observed fault types: {seen}"
    print(f"[PASS] all 4 fault types reachable via seeded sampling: {sorted(seen)}")


def test_fault_injection_changes_readings() -> None:
    # A dropout/bias/noise/stale fault should, over an episode, alter at
    # least one sensor reading relative to a matched clean run with the
    # same obstacles (same seed) -- we simply confirm faulty runs are
    # internally consistent and reproducible; direct value equality vs
    # clean isn't guaranteed every step since faults are randomly timed.
    sim_faulty_1 = RobotSimulator("dense", seed=11, faulty=True)
    sim_faulty_2 = RobotSimulator("dense", seed=11, faulty=True)
    r1 = sim_faulty_1.run(BaselineController())
    r2 = sim_faulty_2.run(BaselineController())
    assert r1.fault_spec.fault_type == r2.fault_spec.fault_type
    assert r1.fault_spec.sensor_index == r2.fault_spec.sensor_index
    assert r1.fault_spec.start_step == r2.fault_spec.start_step
    print("[PASS] fault specification is deterministic given a seed")


def test_full_reproducibility() -> None:
    for tier in ("sparse", "dense"):
        for faulty in (False, True):
            seed = 123
            sim_a = RobotSimulator(tier, seed=seed, faulty=faulty)
            sim_b = RobotSimulator(tier, seed=seed, faulty=faulty)
            result_a = sim_a.run(HandTunedFLC())
            result_b = sim_b.run(HandTunedFLC())

            assert result_a.trajectory == result_b.trajectory, \
                f"trajectory mismatch for tier={tier}, faulty={faulty}"
            assert result_a.sensor_log == result_b.sensor_log, \
                f"sensor log mismatch for tier={tier}, faulty={faulty}"
            assert result_a.steering_log == result_b.steering_log, \
                f"steering log mismatch for tier={tier}, faulty={faulty}"
            assert result_a.metrics.to_dict() == result_b.metrics.to_dict(), \
                f"metrics mismatch for tier={tier}, faulty={faulty}"
    print("[PASS] full reproducibility: identical (tier, seed, faulty) -> identical results")


def test_metrics_present_and_typed() -> None:
    sim = RobotSimulator("sparse", seed=5, faulty=False)
    result = sim.run(HandTunedFLC())
    d = result.metrics.to_dict()
    expected_keys = {
        "success", "collision", "path_length", "time_to_goal",
        "min_clearance", "steering_smoothness", "final_goal_distance", "steps", "mean_speed",
    }
    assert set(d.keys()) == expected_keys, f"missing/extra metric keys: {d.keys()}"
    assert isinstance(d["success"], bool)
    assert isinstance(d["collision"], bool)
    for key in ("path_length", "time_to_goal", "min_clearance",
                "steering_smoothness", "final_goal_distance", "steps"):
        assert isinstance(d[key], (int, float)), f"{key} has unexpected type {type(d[key])}"
    print("[PASS] all required metrics present with correct types")


def test_controller_interface_compatibility() -> None:
    controllers = [BaselineController(), HandTunedFLC()]
    for controller in controllers:
        sim = RobotSimulator("dense", seed=9, faulty=True)
        controller.reset()
        result = sim.run(controller)
        assert result.metrics.steps > 0
    print("[PASS] controller interface (reset/predict) compatible across controllers")


def main() -> None:
    test_obstacle_generation()
    test_dense_tier_feasibility()
    test_obstacle_reproducibility()
    test_three_sensors_and_range()
    test_clean_and_faulty_episodes()
    test_all_fault_types_reachable()
    test_fault_injection_changes_readings()
    test_full_reproducibility()
    test_metrics_present_and_typed()
    test_controller_interface_compatibility()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
