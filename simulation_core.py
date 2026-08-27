"""
simulation_core.py

2D differential-drive mobile robot simulator with a fan of ray-cast range
sensors, static rectangular obstacles, and a sensor fault-injection layer.

The controller picks both a steering angle and a speed each step, and the
robot's motion is integrated in short slices so it cannot pass through a
thin obstacle between one step and the next.

All randomness is drawn from dedicated, explicitly-seeded
`numpy.random.default_rng` generators. No global `np.random.*` calls are
used anywhere in this module, so identical (tier, seed, faulty) inputs
always reproduce identical obstacles, fault specifications, sensor noise,
trajectories, and metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

WORLD_SIZE: float = 100.0
START_POS: Tuple[float, float] = (8.0, 8.0)
GOAL_POS: Tuple[float, float] = (92.0, 92.0)
GOAL_RADIUS: float = 4.0
ROBOT_RADIUS: float = 1.5

SENSOR_RANGE: float = 100.0

# Sensor fan, ordered left to right so that neighbouring indices are
# neighbouring rays. Index 3 (0 deg) looks straight ahead.
#
# This used to be three rays at (0, +45, -45). That left a 45-degree gap on
# each side of the nose and nothing at all past 45 degrees, so an obstacle
# sitting just off the shoulder was invisible until the robot was already
# touching it. Logging the readings at the moment of impact made it obvious:
# the front ray still read ~37 units clear while a side ray read under 3.
# The robot was not driving into things head on, it was scraping them.
#
# Eleven rays at 15-degree spacing over the forward half-plane. The spacing
# matters as much as the count: to work out whether it can fit through a
# gap the robot needs two rays either side of it, and at 45 degrees apart
# it never got them.
SENSOR_ANGLES_DEG: Tuple[float, ...] = tuple(float(a) for a in range(75, -76, -15))
N_SENSORS: int = len(SENSOR_ANGLES_DEG)
FRONT_INDEX: int = SENSOR_ANGLES_DEG.index(0.0)
SENSOR_NAMES: Tuple[str, ...] = tuple(
    "front" if a == 0.0 else f"{'left' if a > 0 else 'right'}{abs(int(a))}"
    for a in SENSOR_ANGLES_DEG
)

MAX_STEER_DEG: float = 45.0
STEER_GAIN: float = 0.3            # commanded degrees -> actual heading change

# Speed is now something the controller chooses, not a fixed constant. The
# turn rate per step does not depend on speed, so going slower tightens the
# turning circle: at 2.0 units/step the robot needs ~8.5 units to come
# around, at 0.4 it needs ~1.7. Being able to slow down is what lets it get
# out of a corner it would previously have ploughed into.
ROBOT_SPEED: float = 2.0           # nominal / default speed
# A differential-drive robot can spin on the spot -- that is the defining
# thing about the platform -- and the floor here used to be 0.4, which
# quietly forbade it. It showed up in the collision traces: five steps
# before impact the robot still had a heading with 47 units of clear space,
# but it was creeping forward at 0.4 the whole time it was turning, and it
# covered the last four units before it finished the turn. Letting it stop
# and rotate removed essentially every collision.
MIN_SPEED: float = 0.0
MAX_SPEED: float = 3.0
COLLISION_SUBSTEP: float = 0.4     # move in slices no longer than this

DT: float = 1.0                    # one control step == one timestep
MAX_STEPS: int = 400

BASELINE_NOISE_STD: float = 0.5    # small Gaussian noise always present ("clean" mode)


# --------------------------------------------------------------------------
# Obstacles
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Obstacle:
    """Axis-aligned rectangular obstacle."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


# BUGFIX (Blueprint v2, Section 2.1 - Dense-Tier Feasibility Audit):
# the original generator only rejected an obstacle for overlapping the
# start/goal regions -- it never checked obstacle-vs-obstacle spacing.
# An audit (debug_controllers.py --audit-feasibility) showed the dense
# tier's minimum obstacle-to-obstacle gap was ~0.0 world units in 100%
# of sampled seeds (obstacles routinely overlapped or abutted each
# other), forming solid unpassable clusters/walls regardless of
# controller quality -- consistent with the reactive baseline
# "winning" dense mode only by accident (getting lucky on open seeds)
# while every other controller scored 0%.
#
# Fix: (a) enforce a minimum obstacle-to-obstacle clearance of
# MIN_OBSTACLE_GAP = 2.5x the robot diameter (matching the blueprint's
# ~2.5x safety-margin rule of thumb), and (b) relax the dense tier's
# obstacle count from 15-25 down to DENSE_OBSTACLE_RANGE so generation
# reliably converges within MAX_GEN_ATTEMPTS while remaining
# meaningfully denser than the sparse tier. SENSOR_RANGE (100) already
# comfortably exceeds 1.5x the minimum passable gap, so sensor range
# was not adjusted.
MIN_OBSTACLE_GAP: float = 2.5 * 2.0 * ROBOT_RADIUS   # ~7.5 world units
MAX_GEN_ATTEMPTS: int = 3000
# The dense tier is supposed to be the harder one. Measured over 60 seeds it
# was covering 5.5% of the world against the sparse tier's 6.8% -- sparse
# has fewer obstacles but much bigger ones, so "dense" was actually the
# roomier of the two and the tier comparison meant nothing. 22-30 puts it at
# about 9.2%, comfortably denser, and generation still places everything it
# is asked for. MIN_OBSTACLE_GAP is still enforced, so every gap remains
# wide enough to drive through -- denser, not impassable.
DENSE_OBSTACLE_RANGE: Tuple[int, int] = (22, 30)
SPARSE_OBSTACLE_RANGE: Tuple[int, int] = (5, 8)


def generate_obstacles(tier: str, seed: int) -> List[Obstacle]:
    """
    Deterministically generate obstacles for a given tier and seed.

    tier: "sparse" (5-8 obstacles) or "dense" (22-30 obstacles, see
    MIN_OBSTACLE_GAP fix note above). Obstacles never overlap the
    start/goal regions, and never sit closer than MIN_OBSTACLE_GAP to
    another obstacle, guaranteeing every gap in the layout is wide
    enough for the robot (with its turning radius) to physically pass.
    """
    if tier not in ("sparse", "dense"):
        raise ValueError(f"Unknown tier '{tier}'. Must be 'sparse' or 'dense'.")

    rng = np.random.default_rng(seed)

    if tier == "sparse":
        n_obstacles = int(rng.integers(SPARSE_OBSTACLE_RANGE[0], SPARSE_OBSTACLE_RANGE[1] + 1))
        size_range = (6.0, 14.0)
    else:
        n_obstacles = int(rng.integers(DENSE_OBSTACLE_RANGE[0], DENSE_OBSTACLE_RANGE[1] + 1))
        size_range = (4.0, 8.0)

    forbidden_radius = 10.0
    obstacles: List[Obstacle] = []
    attempts = 0

    while len(obstacles) < n_obstacles and attempts < MAX_GEN_ATTEMPTS:
        attempts += 1
        w = float(rng.uniform(*size_range))
        h = float(rng.uniform(*size_range))
        cx = float(rng.uniform(5.0, WORLD_SIZE - 5.0))
        cy = float(rng.uniform(5.0, WORLD_SIZE - 5.0))

        x_min, x_max = cx - w / 2.0, cx + w / 2.0
        y_min, y_max = cy - h / 2.0, cy + h / 2.0
        x_min, y_min = max(0.0, x_min), max(0.0, y_min)
        x_max, y_max = min(WORLD_SIZE, x_max), min(WORLD_SIZE, y_max)

        candidate = Obstacle(x_min, y_min, x_max, y_max)

        if _dist_point_to_rect(START_POS[0], START_POS[1], candidate) < forbidden_radius:
            continue
        if _dist_point_to_rect(GOAL_POS[0], GOAL_POS[1], candidate) < forbidden_radius:
            continue
        if any(_dist_rect_to_rect(candidate, o) < MIN_OBSTACLE_GAP for o in obstacles):
            continue

        obstacles.append(candidate)

    return obstacles


def _dist_point_to_rect(px: float, py: float, rect: Obstacle) -> float:
    dx = max(rect.x_min - px, 0.0, px - rect.x_max)
    dy = max(rect.y_min - py, 0.0, py - rect.y_max)
    return math.hypot(dx, dy)


def _dist_rect_to_rect(a: Obstacle, b: Obstacle) -> float:
    dx = max(a.x_min - b.x_max, b.x_min - a.x_max, 0.0)
    dy = max(a.y_min - b.y_max, b.y_min - a.y_max, 0.0)
    return math.hypot(dx, dy)


# --------------------------------------------------------------------------
# Fault specification and injection
# --------------------------------------------------------------------------

FAULT_TYPES: Tuple[str, ...] = ("dropout", "bias", "noise_spike", "stale")


@dataclass
class FaultSpec:
    """A single fault instance applied during one episode."""
    fault_type: str
    sensor_index: int
    start_step: int
    duration: int
    magnitude: float


def sample_fault_spec(seed: int) -> FaultSpec:
    """
    Deterministically sample one fault (type, sensor, timing window,
    magnitude) for a "faulty" episode. Using the same seed always
    produces the same fault instance, so all controllers evaluated
    with that seed face an identical fault.
    """
    rng = np.random.default_rng(seed)
    fault_type = FAULT_TYPES[int(rng.integers(0, len(FAULT_TYPES)))]
    sensor_index = int(rng.integers(0, len(SENSOR_NAMES)))
    # Most episodes resolve (success or collision) well before MAX_STEPS, so
    # faults must start early to actually fall within an episode's decisive
    # window rather than after the outcome is already settled.
    start_step = int(rng.integers(2, 40))

    if fault_type == "dropout":
        duration = int(rng.integers(40, 90))
        magnitude = 0.0
    elif fault_type == "bias":
        duration = MAX_STEPS - start_step          # persists for rest of episode
        magnitude = float(rng.uniform(20.0, 45.0)) * float(rng.choice([-1.0, 1.0]))
    elif fault_type == "noise_spike":
        duration = int(rng.integers(40, 90))
        magnitude = float(rng.uniform(15.0, 30.0))   # elevated std
    else:  # stale
        duration = int(rng.integers(40, 90))
        magnitude = float(rng.integers(5, 15))      # delay in steps      # delay in steps   # delay in steps

    return FaultSpec(
        fault_type=fault_type,
        sensor_index=sensor_index,
        start_step=start_step,
        duration=duration,
        magnitude=magnitude,
    )


def sample_fault_spec_typed(seed: int, fault_type: Optional[str] = None,
                             severity_range: Optional[Tuple[float, float]] = None) -> FaultSpec:
    """
    Like sample_fault_spec, but allows pinning the fault type and/or the
    magnitude range explicitly. Used for the per-fault-type breakdown and
    severity-sweep experiments required by Blueprint v2 Sec 5.1 (a plain
    `sample_fault_spec(seed)` only gives a *random* fault per seed, which
    is right for the original clean/faulty grid but not for "give me a
    dropout fault" or "give me a severity-3 noise-spike fault").
    """
    rng = np.random.default_rng(seed)
    ftype = fault_type if fault_type is not None else FAULT_TYPES[int(rng.integers(0, len(FAULT_TYPES)))]
    sensor_index = int(rng.integers(0, len(SENSOR_NAMES)))
    start_step = int(rng.integers(2, 40))

    if ftype == "dropout":
        duration = int(rng.integers(40, 90))
        magnitude = 0.0
    elif ftype == "bias":
        duration = MAX_STEPS - start_step
        lo, hi = severity_range if severity_range else (20.0, 45.0)
        magnitude = float(rng.uniform(lo, hi)) * float(rng.choice([-1.0, 1.0]))
    elif ftype == "noise_spike":
        duration = int(rng.integers(40, 90))
        lo, hi = severity_range if severity_range else (15.0, 30.0)
        magnitude = float(rng.uniform(lo, hi))
    elif ftype == "stale":
        duration = int(rng.integers(40, 90))
        magnitude = float(rng.integers(5, 15))
    else:
        raise ValueError(f"Unknown fault_type '{ftype}'")

    return FaultSpec(fault_type=ftype, sensor_index=sensor_index,
                      start_step=start_step, duration=duration, magnitude=magnitude)


class FaultInjector:
    """
    Wraps raw sensor readings. In clean mode, applies only small baseline
    Gaussian noise. In faulty mode, additionally applies exactly one
    FaultSpec on top of the baseline noise.

    `spec_override` lets callers (experiment grid, fault-detector dataset
    generation) pin a specific fault type/severity instead of the fully
    random `sample_fault_spec`. `manual` mode (used by the interactive
    Pygame visualizer) starts with no fault and lets the caller inject one
    live at an arbitrary step via `trigger_manual_fault`.
    """

    def __init__(self, faulty: bool, seed: int,
                 spec_override: Optional[FaultSpec] = None,
                 manual: bool = False):
        self.faulty = faulty
        self._rng = np.random.default_rng(seed)
        if manual:
            self.spec: Optional[FaultSpec] = None
        elif spec_override is not None:
            self.spec = spec_override
        else:
            self.spec = sample_fault_spec(seed) if faulty else None
        self._history: List[np.ndarray] = []  # for stale-reading delay

    def trigger_manual_fault(self, fault_type: str, sensor_index: int,
                              step: int, duration: int = 10_000,
                              magnitude: Optional[float] = None) -> None:
        """Inject a fault starting at `step`, live (for the interactive demo)."""
        if magnitude is None:
            magnitude = {"dropout": 0.0, "bias": 30.0, "noise_spike": 20.0, "stale": 8.0}[fault_type]
        self.spec = FaultSpec(fault_type=fault_type, sensor_index=sensor_index,
                               start_step=step, duration=duration, magnitude=magnitude)

    def clear_fault(self) -> None:
        self.spec = None

    def apply(self, raw_readings: np.ndarray, step: int) -> np.ndarray:
        """Return the (possibly faulted) sensor readings for this step."""
        readings = raw_readings.copy()
        readings += self._rng.normal(0.0, BASELINE_NOISE_STD, size=readings.shape)
        readings = np.clip(readings, 0.0, SENSOR_RANGE)

        self._history.append(raw_readings.copy())

        if self.spec is None:
            return readings

        s = self.spec
        active = s.start_step <= step < s.start_step + s.duration
        if not active:
            return readings

        idx = s.sensor_index
        if s.fault_type == "dropout":
            frozen_value = self._history[s.start_step][idx] if s.start_step < len(self._history) else readings[idx]
            readings[idx] = frozen_value
        elif s.fault_type == "bias":
            readings[idx] = np.clip(readings[idx] + s.magnitude, 0.0, SENSOR_RANGE)
        elif s.fault_type == "noise_spike":
            extra = self._rng.normal(0.0, s.magnitude)
            readings[idx] = np.clip(readings[idx] + extra, 0.0, SENSOR_RANGE)
        elif s.fault_type == "stale":
            delay = max(1, int(s.magnitude))
            hist_idx = max(0, len(self._history) - 1 - delay)
            readings[idx] = self._history[hist_idx][idx]

        return readings


# --------------------------------------------------------------------------
# Ray-casting sensor model
# --------------------------------------------------------------------------

def _ray_rect_intersection(px: float, py: float, dx: float, dy: float,
                            rect: Obstacle, max_range: float) -> Optional[float]:
    """Return distance to intersection of ray with rect, or None."""
    t_min, t_max = 0.0, max_range

    for origin, direction, lo, hi in (
        (px, dx, rect.x_min, rect.x_max),
        (py, dy, rect.y_min, rect.y_max),
    ):
        if abs(direction) < 1e-12:
            if origin < lo or origin > hi:
                return None
        else:
            t1 = (lo - origin) / direction
            t2 = (hi - origin) / direction
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None

    if t_min < 0:
        return None
    return t_min


def cast_ray(x: float, y: float, angle_rad: float, obstacles: List[Obstacle],
             max_range: float = SENSOR_RANGE) -> float:
    """Cast a single ray and return distance to the nearest obstacle or wall."""
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)

    best = max_range

    # World boundary walls act as obstacles.
    for lo, hi, origin, direction in (
        (0.0, WORLD_SIZE, x, dx),
        (0.0, WORLD_SIZE, y, dy),
    ):
        pass  # handled implicitly by clipping below

    # Wall distances (world is [0, WORLD_SIZE]^2)
    if dx > 1e-12:
        t = (WORLD_SIZE - x) / dx
        best = min(best, t)
    elif dx < -1e-12:
        t = (0.0 - x) / dx
        best = min(best, t)
    if dy > 1e-12:
        t = (WORLD_SIZE - y) / dy
        best = min(best, t)
    elif dy < -1e-12:
        t = (0.0 - y) / dy
        best = min(best, t)

    for obs in obstacles:
        dist = _ray_rect_intersection(x, y, dx, dy, obs, max_range)
        if dist is not None and dist < best:
            best = dist

    return max(0.0, min(best, max_range))


def get_sensor_readings(x: float, y: float, heading_rad: float,
                         obstacles: List[Obstacle]) -> np.ndarray:
    """Return raw range readings for the whole fan, in [0, SENSOR_RANGE]."""
    readings = np.zeros(N_SENSORS, dtype=np.float64)
    for i, angle_deg in enumerate(SENSOR_ANGLES_DEG):
        angle_rad = heading_rad + math.radians(angle_deg)
        readings[i] = cast_ray(x, y, angle_rad, obstacles)
    return readings


def get_goal_observation(x: float, y: float, heading_rad: float) -> Tuple[float, float]:
    """
    Goal-relative observation: (goal_distance, goal_angle_deg).

    goal_distance is capped at SENSOR_RANGE, matching the sensor readings'
    scale (so all 5 observation components share one [0, 100]-ish range
    rather than the controller having to learn/hand-tune a second scale).
    goal_angle_deg is the signed angle from the robot's current heading to
    the goal direction, in degrees, using the SAME sign convention already
    validated for steering and the left/right sensors (positive = counter-
    clockwise = "goal is to the left"; see fuzzy_handtuned.py's sign-
    convention check and validate_simulator.py): a controller can use
    goal_angle_deg directly as a "steer this many degrees to head straight
    at the goal" signal without any sign translation.

    Assumed to come from the robot's own localization/odometry (a
    standard assumption in this class of navigation problem), NOT a
    range sensor -- so it is never passed through FaultInjector and
    fault-injection/detection remain scoped to the range readings only,
    unchanged.
    """
    dx = GOAL_POS[0] - x
    dy = GOAL_POS[1] - y
    goal_distance = min(SENSOR_RANGE, math.hypot(dx, dy))
    goal_heading = math.atan2(dy, dx)
    goal_angle_rad = math.atan2(math.sin(goal_heading - heading_rad),
                                 math.cos(goal_heading - heading_rad))
    return goal_distance, math.degrees(goal_angle_rad)


OBSERVATION_DIM: int = N_SENSORS + 2
GOAL_DIST_INDEX: int = N_SENSORS
GOAL_ANGLE_INDEX: int = N_SENSORS + 1


def build_full_observation(sensor_readings: np.ndarray, x: float, y: float,
                            heading_rad: float) -> np.ndarray:
    """The complete observation handed to every controller's predict():
    all N_SENSORS range readings, then goal_distance and goal_angle_deg."""
    goal_dist, goal_angle = get_goal_observation(x, y, heading_rad)
    return np.concatenate([
        np.asarray(sensor_readings, dtype=np.float64).ravel(),
        np.array([goal_dist, goal_angle], dtype=np.float64),
    ])


def unpack_action(action) -> Tuple[float, float]:
    """
    Controllers return either a bare steering angle or (steering, speed).

    Everything in this project returns the pair now, but accepting a plain
    float keeps the older sensor-only controllers runnable without a
    wrapper, which is handy when checking one against the other.
    """
    if isinstance(action, (tuple, list, np.ndarray)):
        steer, speed = float(action[0]), float(action[1])
    else:
        steer, speed = float(action), ROBOT_SPEED
    steer = float(np.clip(steer, -MAX_STEER_DEG, MAX_STEER_DEG))
    speed = float(np.clip(speed, MIN_SPEED, MAX_SPEED))
    return steer, speed


# --------------------------------------------------------------------------
# Episode metrics
# --------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    success: bool = False
    collision: bool = False
    path_length: float = 0.0
    time_to_goal: int = 0
    min_clearance: float = SENSOR_RANGE
    steering_smoothness: float = 0.0   # mean abs steering-angle change (deg)
    final_goal_distance: float = 0.0
    steps: int = 0
    mean_speed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "collision": self.collision,
            "path_length": self.path_length,
            "time_to_goal": self.time_to_goal,
            "min_clearance": self.min_clearance,
            "steering_smoothness": self.steering_smoothness,
            "final_goal_distance": self.final_goal_distance,
            "steps": self.steps,
            "mean_speed": self.mean_speed,
        }


@dataclass
class EpisodeResult:
    metrics: EpisodeMetrics
    trajectory: List[Tuple[float, float]]
    sensor_log: List[Tuple[float, ...]]
    steering_log: List[float]
    fault_spec: Optional[FaultSpec]
    clean_sensor_log: List[Tuple[float, ...]] = field(default_factory=list)
    full_observation_log: List[Tuple[float, ...]] = field(default_factory=list)
    speed_log: List[float] = field(default_factory=list)


# --------------------------------------------------------------------------
# Robot simulation loop
# --------------------------------------------------------------------------

class RobotSimulator:
    """
    Runs a single differential-drive robot episode against a controller.

    The controller must expose:
        reset() -> None
        predict(sensor_readings: np.ndarray) -> float   # steering angle, deg
    """

    def __init__(self, tier: str, seed: int, faulty: bool,
                 spec_override: Optional[FaultSpec] = None,
                 manual_fault: bool = False):
        self.tier = tier
        self.seed = seed
        self.faulty = faulty
        self.obstacles = generate_obstacles(tier, seed)
        self.fault_injector = FaultInjector(faulty=faulty, seed=seed,
                                             spec_override=spec_override,
                                             manual=manual_fault)

    def _clearance(self, x: float, y: float) -> float:
        """Distance from the robot's edge to the nearest obstacle or wall.

        The old measure was the smallest of the range readings, which is
        only the distance along whichever way a ray happened to point. With
        a wider fan that got closer to the truth but it still misses
        anything sitting beside the robot. This is the actual distance, so
        the safety objective is scoring what it claims to score.
        """
        nearest = min(x, y, WORLD_SIZE - x, WORLD_SIZE - y)
        for obs in self.obstacles:
            nx = min(max(x, obs.x_min), obs.x_max)
            ny = min(max(y, obs.y_min), obs.y_max)
            nearest = min(nearest, math.hypot(x - nx, y - ny))
        return max(0.0, nearest - ROBOT_RADIUS)

    def _move_checked(self, x: float, y: float, heading: float,
                       distance: float) -> Tuple[float, float, bool]:
        """Advance along `heading` in short slices, stopping at a collision.

        A single jump of up to 3 units with a robot only 3 units across can
        pass clean through a thin obstacle without ever landing inside it,
        which reads as a success that never happened. Splitting the step
        makes the check continuous enough that it cannot be skipped over.
        """
        n = max(1, int(math.ceil(distance / COLLISION_SUBSTEP)))
        step_len = distance / n
        dx, dy = math.cos(heading) * step_len, math.sin(heading) * step_len
        for _ in range(n):
            x, y = x + dx, y + dy
            if self._collides(x, y):
                return x, y, True
        return x, y, False

    def _collides(self, x: float, y: float) -> bool:
        if x - ROBOT_RADIUS < 0 or x + ROBOT_RADIUS > WORLD_SIZE:
            return True
        if y - ROBOT_RADIUS < 0 or y + ROBOT_RADIUS > WORLD_SIZE:
            return True
        for obs in self.obstacles:
            nearest_x = min(max(x, obs.x_min), obs.x_max)
            nearest_y = min(max(y, obs.y_min), obs.y_max)
            if math.hypot(x - nearest_x, y - nearest_y) < ROBOT_RADIUS:
                return True
        return False

    def run(self, controller, max_steps: int = MAX_STEPS) -> EpisodeResult:
        controller.reset()

        x, y = START_POS
        heading = math.atan2(GOAL_POS[1] - y, GOAL_POS[0] - x)

        trajectory: List[Tuple[float, float]] = [(x, y)]
        sensor_log: List[Tuple[float, ...]] = []
        clean_sensor_log: List[Tuple[float, ...]] = []
        steering_log: List[float] = []
        speed_log: List[float] = []
        full_observation_log: List[Tuple[float, ...]] = []

        metrics = EpisodeMetrics()
        prev_steer = 0.0
        smoothness_accum = 0.0
        path_length = 0.0
        min_clearance = SENSOR_RANGE

        step = 0
        for step in range(max_steps):
            raw = get_sensor_readings(x, y, heading, self.obstacles)
            observed = self.fault_injector.apply(raw, step)
            sensor_log.append(tuple(observed.tolist()))
            # Kept so the detector's training labels can tell whether a fault
            # was actually changing what the robot saw at that moment.
            clean_sensor_log.append(tuple(raw.tolist()))

            min_clearance = min(min_clearance, self._clearance(x, y))

            full_obs = build_full_observation(observed, x, y, heading)
            full_observation_log.append(tuple(full_obs.tolist()))

            steer_deg, speed = unpack_action(controller.predict(full_obs))
            steering_log.append(steer_deg)
            speed_log.append(speed)

            smoothness_accum += abs(steer_deg - prev_steer)
            prev_steer = steer_deg

            heading += math.radians(steer_deg) * STEER_GAIN
            heading = math.atan2(math.sin(heading), math.cos(heading))

            prev_x, prev_y = x, y
            x, y, hit = self._move_checked(x, y, heading, speed * DT)
            path_length += math.hypot(x - prev_x, y - prev_y)
            trajectory.append((x, y))

            if hit:
                metrics.collision = True
                break

            dist_to_goal = math.hypot(GOAL_POS[0] - x, GOAL_POS[1] - y)
            if dist_to_goal <= GOAL_RADIUS:
                metrics.success = True
                break

        metrics.path_length = path_length
        metrics.time_to_goal = step + 1 if metrics.success else max_steps
        metrics.min_clearance = min_clearance
        metrics.steering_smoothness = smoothness_accum / max(1, len(steering_log))
        metrics.final_goal_distance = math.hypot(GOAL_POS[0] - x, GOAL_POS[1] - y)
        metrics.steps = step + 1
        metrics.mean_speed = float(np.mean(speed_log)) if speed_log else 0.0

        return EpisodeResult(
            metrics=metrics,
            trajectory=trajectory,
            sensor_log=sensor_log,
            steering_log=steering_log,
            fault_spec=self.fault_injector.spec,
            full_observation_log=full_observation_log,
            speed_log=speed_log,
            clean_sensor_log=clean_sensor_log,
        )

    def step_iter(self, controller, max_steps: int = MAX_STEPS):
        """
        Generator version of `run`, used by the interactive Pygame
        visualizer so it can render one frame per step, support
        pause/step/restart, and let the user trigger a live fault via
        `self.fault_injector.trigger_manual_fault(...)` between frames.
        Yields a dict snapshot of simulator state after each step.
        """
        controller.reset()
        x, y = START_POS
        heading = math.atan2(GOAL_POS[1] - y, GOAL_POS[0] - x)
        prev_steer = 0.0
        path_length = 0.0
        min_clearance = SENSOR_RANGE
        done = False
        outcome = None

        for step in range(max_steps):
            raw = get_sensor_readings(x, y, heading, self.obstacles)
            observed = self.fault_injector.apply(raw, step)
            min_clearance = min(min_clearance, self._clearance(x, y))

            full_obs = build_full_observation(observed, x, y, heading)
            steer_deg, speed = unpack_action(controller.predict(full_obs))

            heading += math.radians(steer_deg) * STEER_GAIN
            heading = math.atan2(math.sin(heading), math.cos(heading))

            prev_x, prev_y = x, y
            x, y, hit = self._move_checked(x, y, heading, speed * DT)
            path_length += math.hypot(x - prev_x, y - prev_y)

            if hit:
                done, outcome = True, "collision"
            else:
                dist_to_goal = math.hypot(GOAL_POS[0] - x, GOAL_POS[1] - y)
                if dist_to_goal <= GOAL_RADIUS:
                    done, outcome = True, "success"

            yield {
                "step": step,
                "x": x, "y": y, "heading": heading,
                "raw_sensors": raw, "observed_sensors": observed,
                "steer_deg": steer_deg, "speed": speed,
                "fault_spec": self.fault_injector.spec,
                "path_length": path_length,
                "min_clearance": min_clearance,
                "done": done, "outcome": outcome,
            }
            prev_steer = steer_deg
            if done:
                return
        yield {"step": max_steps, "x": x, "y": y, "heading": heading,
               "done": True, "outcome": "timeout", "path_length": path_length,
               "min_clearance": min_clearance, "fault_spec": self.fault_injector.spec,
               "raw_sensors": raw, "observed_sensors": observed, "steer_deg": prev_steer}
