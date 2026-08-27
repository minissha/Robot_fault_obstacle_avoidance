"""
make_demo.py

Renders an episode to an animated GIF: the robot driving, its sensor rays,
and a sensor fault injected part-way through so you can watch what the
fault-aware controller does about it.

The Pygame viewer (viz/simulator_view.py) is the interactive version of
this and needs a screen. This one writes a file, so it works over SSH, in
CI, or anywhere else without a display -- which is what you need if the
output is meant to be attached to something.

Run from the project root:
    python -m viz.make_demo                       # fault-aware, bias fault
    python -m viz.make_demo --controller flc --fault stale --blind
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from config import DEMO_DIR, FAULT_DETECTOR_PATH, get_eval_trial_seeds
from simulation_core import (
    GOAL_POS, GOAL_RADIUS, N_SENSORS, ROBOT_RADIUS, SENSOR_ANGLES_DEG,
    SENSOR_NAMES, START_POS, WORLD_SIZE, RobotSimulator, sample_fault_spec_typed,
)
from controllers.baseline import BaselineController
from controllers.fuzzy_handtuned import HandTunedFLC


def build_controller(kind: str, blind: bool):
    base = BaselineController() if kind == "baseline" else HandTunedFLC()
    if blind:
        return base, None
    if not os.path.exists(FAULT_DETECTOR_PATH):
        print(f"[warn] {FAULT_DETECTOR_PATH} missing -- running fault-blind.")
        return base, None
    from controllers.fault_detector import FaultDetector
    from controllers.fault_aware import FaultAwareController
    wrapped = FaultAwareController(base, FaultDetector.load(FAULT_DETECTOR_PATH))
    return wrapped, wrapped


def render(kind: str, tier: str, seed: int, fault: str, blind: bool,
           out_path: str, fps: int) -> str:
    controller, aware = build_controller(kind, blind)
    spec = sample_fault_spec_typed(seed, fault_type=fault)
    sim = RobotSimulator(tier, seed=seed, faulty=True, spec_override=spec)
    result = sim.run(controller)

    traj = np.array(result.trajectory)
    readings = np.array(result.sensor_log)
    trust = np.array(aware.trust_log) if aware is not None and aware.trust_log else None
    n_frames = len(result.sensor_log)

    fig, (ax, ax_info) = plt.subplots(
        1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})

    for obs in sim.obstacles:
        ax.add_patch(plt.Rectangle((obs.x_min, obs.y_min),
                                    obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                                    facecolor="0.78", edgecolor="0.45", zorder=1))
    ax.add_patch(plt.Circle(GOAL_POS, GOAL_RADIUS, facecolor="tab:green",
                             alpha=0.35, zorder=1))
    ax.plot(*START_POS, "ko", markersize=5, zorder=2)
    ax.set_xlim(0, WORLD_SIZE); ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    trail, = ax.plot([], [], "-", color="tab:blue", linewidth=1.6, zorder=3)
    body = plt.Circle(START_POS, ROBOT_RADIUS, facecolor="tab:blue",
                       edgecolor="black", zorder=5)
    ax.add_patch(body)
    rays = [ax.plot([], [], "-", linewidth=1.0, alpha=0.8, zorder=4)[0]
            for _ in range(N_SENSORS)]

    ax_info.axis("off")
    text = ax_info.text(0.0, 0.98, "", va="top", family="monospace", fontsize=9)

    def heading_at(i):
        if i + 1 < len(traj):
            d = traj[i + 1] - traj[i]
            if np.hypot(*d) > 1e-9:
                return math.atan2(d[1], d[0])
        return math.atan2(GOAL_POS[1] - traj[i][1], GOAL_POS[0] - traj[i][0])

    def frame(i):
        x, y = traj[i]
        body.center = (x, y)
        trail.set_data(traj[:i + 1, 0], traj[:i + 1, 1])
        h = heading_at(i)
        active = spec.start_step <= i < spec.start_step + spec.duration

        for k, ang in enumerate(SENSOR_ANGLES_DEG):
            d = readings[i][k]
            a = h + math.radians(ang)
            rays[k].set_data([x, x + d * math.cos(a)], [y, y + d * math.sin(a)])
            if k == spec.sensor_index and active:
                rays[k].set_color("tab:red"); rays[k].set_linewidth(2.0)
            elif trust is not None and i < len(trust) and trust[i][k] < 0.9:
                rays[k].set_color("tab:orange"); rays[k].set_linewidth(1.6)
            else:
                rays[k].set_color("tab:cyan"); rays[k].set_linewidth(1.0)

        lines = [
            f"controller : {kind}{'' if blind else ' + fault-aware'}",
            f"tier/seed  : {tier} / {seed}",
            f"step       : {i + 1} / {n_frames}",
            f"speed      : {result.speed_log[i]:.2f} units/step",
            f"steering   : {result.steering_log[i]:+.1f} deg",
            "",
            f"fault      : {fault} on {SENSOR_NAMES[spec.sensor_index]}",
            f"             {'ACTIVE' if active else 'not active'} "
            f"(steps {spec.start_step}-{spec.start_step + spec.duration})",
            "",
        ]
        if trust is not None and i < len(trust):
            lines.append("per-ray trust (1.00 = believed):")
            for k in range(N_SENSORS):
                bar = "#" * int(round(trust[i][k] * 18))
                lines.append(f"  {SENSOR_NAMES[k]:>8s} {trust[i][k]:.2f} {bar}")
        else:
            lines.append("(fault-blind: no detector, readings used as-is)")
        text.set_text("\n".join(lines))

        outcome = ("reached the goal" if result.metrics.success
                   else "collided" if result.metrics.collision else "ran out of time")
        ax.set_title(f"{'' if i + 1 < n_frames else 'FINISHED - '}{outcome}"
                      if i + 1 == n_frames else "Fault-aware navigation demo")
        return [trail, body, text, *rays]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    anim = FuncAnimation(fig, frame, frames=n_frames, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved {out_path} ({n_frames} frames at {fps} fps "
          f"= {n_frames / fps:.1f}s, outcome: "
          f"{'success' if result.metrics.success else 'failure'})")
    return out_path


def find_contrast_seed(kind: str, tier: str, fault: str, n: int = 60):
    """A seed where the fault-blind run fails and the fault-aware one does not.

    That contrast is the whole point of the demo -- same map, same fault,
    same controller underneath, and one of them gets there. Falls back to
    the first seed rather than pretending a contrast exists.
    """
    from controllers.fault_detector import FaultDetector
    from controllers.fault_aware import FaultAwareController

    if not os.path.exists(FAULT_DETECTOR_PATH):
        return get_eval_trial_seeds()[0]
    detector = FaultDetector.load(FAULT_DETECTOR_PATH)
    make_base = BaselineController if kind == "baseline" else HandTunedFLC

    for seed in get_eval_trial_seeds()[:n]:
        spec = sample_fault_spec_typed(seed, fault_type=fault)
        blind = RobotSimulator(tier, seed=seed, faulty=True,
                                spec_override=spec).run(make_base())
        if blind.metrics.success:
            continue
        aware = RobotSimulator(tier, seed=seed, faulty=True, spec_override=spec).run(
            FaultAwareController(make_base(), detector))
        if aware.metrics.success:
            print(f"Using seed {seed}: fault-blind fails here, fault-aware does not.")
            return seed
    print("No seed found where fault-awareness flips the outcome; using the first.")
    return get_eval_trial_seeds()[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller", choices=["baseline", "flc"], default="flc")
    p.add_argument("--tier", choices=["sparse", "dense"], default="dense")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fault", choices=["dropout", "bias", "noise_spike", "stale"],
                    default="bias")
    p.add_argument("--blind", action="store_true",
                    help="run without the fault detector, for a side-by-side")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--auto-seed", action="store_true",
                    help="search for a seed where fault-awareness changes the outcome")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    if a.seed is not None:
        seed = a.seed
    elif a.auto_seed:
        seed = find_contrast_seed(a.controller, a.tier, a.fault)
    else:
        seed = get_eval_trial_seeds()[0]
    name = (f"demo_{a.controller}_{a.tier}_{a.fault}"
            f"{'_blind' if a.blind else '_aware'}.gif")
    render(a.controller, a.tier, seed, a.fault, a.blind,
           a.out or os.path.join(DEMO_DIR, name), a.fps)


if __name__ == "__main__":
    main()
