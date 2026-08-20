"""
viz/simulator_view.py

Blueprint upgrade request #1: interactive 2D visualization (Pygame).

Renders the robot, obstacles, goal, trajectory trail, heading, and all 3
sensor rays/readings; shows controller name, sensor status, fault
type/severity, detector confidence (if a trained fault detector is
available), speed, steering angle, and success/collision state. Supports
pause, restart, single-step, playback-speed adjustment, seed selection,
and live sensor-fault injection (dropout/bias/noise-spike/stale) via
keyboard.

Loosely inspired by the render loop structure used in
reiniscimurs/DRL-robot-navigation (a Pygame/ROS visualizer for a
different DRL navigation project) -- rays-from-robot + trail rendering is
a standard pattern in that family of projects -- but this is an
independent implementation against THIS project's own simulation_core.py
and controllers, not a port of that codebase.

Visualization is intentionally decoupled from simulation logic: this
module only calls RobotSimulator.step_iter(...) and
FaultInjector.trigger_manual_fault(...) from simulation_core.py; it does
not duplicate any physics, sensing, or fault-injection code.

Run from the project root:
    python -m viz.simulator_view
    python -m viz.simulator_view --controller flc --tier dense --seed 7
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

from simulation_core import (
    RobotSimulator, WORLD_SIZE, ROBOT_RADIUS, GOAL_POS, GOAL_RADIUS,
    SENSOR_ANGLES_DEG, SENSOR_RANGE, MAX_STEPS,
)
from controllers.baseline import BaselineController
from controllers.fuzzy_handtuned import HandTunedFLC
from config import FAULT_DETECTOR_PATH

# --------------------------------------------------------------------------
# Layout / colors
# --------------------------------------------------------------------------
WINDOW_W, WINDOW_H = 1100, 820
WORLD_PANEL = 760           # left panel: world render, WORLD_PANEL x WORLD_PANEL px
MARGIN = 20
SCALE = (WORLD_PANEL - 2 * MARGIN) / WORLD_SIZE

BG = (18, 18, 24)
PANEL_BG = (26, 26, 34)
OBSTACLE_COLOR = (90, 90, 110)
ROBOT_COLOR = (80, 200, 255)
GOAL_COLOR = (80, 230, 120)
TRAIL_COLOR = (80, 200, 255, 90)
RAY_OK_COLOR = (90, 220, 140)
RAY_WARN_COLOR = (230, 200, 80)
RAY_FAULT_COLOR = (235, 90, 90)
TEXT_COLOR = (230, 230, 235)
MUTED_TEXT = (150, 150, 160)

CONTROLLERS = {
    "baseline": BaselineController,
    "flc": HandTunedFLC,
}
FAULT_KEYS = {
    pygame.K_1 if pygame else 1: "dropout",
    pygame.K_2 if pygame else 2: "bias",
    pygame.K_3 if pygame else 3: "noise_spike",
    pygame.K_4 if pygame else 4: "stale",
}


def world_to_screen(x: float, y: float):
    sx = MARGIN + x * SCALE
    sy = MARGIN + (WORLD_SIZE - y) * SCALE  # flip Y so +y is "up" on screen
    return int(sx), int(sy)


class SimulatorApp:
    def __init__(self, controller_name: str, tier: str, seed: int):
        if pygame is None:
            raise RuntimeError(
                "pygame is not installed in this environment. Install it with "
                "'pip install pygame' and re-run: python -m viz.simulator_view"
            )
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Fault-Aware Robot Navigation -- Interactive Demo")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_small = pygame.font.SysFont("consolas", 13)
        self.font_big = pygame.font.SysFont("consolas", 20, bold=True)

        self.controller_name = controller_name
        self.tier = tier
        self.seed = seed
        self.paused = False
        self.step_once = False
        self.speed = 1  # simulation steps per rendered frame
        self.detector = None
        if os.path.exists(FAULT_DETECTOR_PATH):
            from controllers.fault_detector import FaultDetector
            self.detector = FaultDetector.load(FAULT_DETECTOR_PATH)

        self._reset_episode()

    def _make_controller(self):
        return CONTROLLERS[self.controller_name]()

    def _reset_episode(self):
        self.sim = RobotSimulator(self.tier, seed=self.seed, faulty=False, manual_fault=True)
        self.controller = self._make_controller()
        if self.detector is not None:
            self.detector.reset()
        self.gen = self.sim.step_iter(self.controller, max_steps=MAX_STEPS)
        self.trail = [self.sim.__dict__.get("_start", (8.0, 8.0))]
        self.trail = []
        self.last_state = None
        self.done = False
        self.detector_labels = ["none"] * 3
        self.detector_confs = [0.0] * 3

    def _advance(self):
        if self.done:
            return
        try:
            state = next(self.gen)
        except StopIteration:
            self.done = True
            return
        self.last_state = state
        self.trail.append((state["x"], state["y"]))
        if self.detector is not None:
            labels, confs = self.detector.push_and_predict(state["observed_sensors"])
            self.detector_labels, self.detector_confs = labels, confs
        if state["done"]:
            self.done = True

    def _handle_keys(self, event) -> None:
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif event.key == pygame.K_RIGHT:
            self.step_once = True
        elif event.key == pygame.K_r:
            self._reset_episode()
        elif event.key == pygame.K_UP:
            self.speed = min(20, self.speed + 1)
        elif event.key == pygame.K_DOWN:
            self.speed = max(1, self.speed - 1)
        elif event.key == pygame.K_c:
            self.controller_name = "flc" if self.controller_name == "baseline" else "baseline"
            self._reset_episode()
        elif event.key == pygame.K_t:
            self.tier = "dense" if self.tier == "sparse" else "sparse"
            self._reset_episode()
        elif event.key == pygame.K_n:
            self.seed += 1
            self._reset_episode()
        elif event.key == pygame.K_p:
            self.seed = max(0, self.seed - 1)
            self._reset_episode()
        elif event.key in FAULT_KEYS and not self.done:
            fault_type = FAULT_KEYS[event.key]
            sensor_idx = 0  # front sensor by default; Shift+number picks left/right below
            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_SHIFT:
                sensor_idx = 1  # left
            elif mods & pygame.KMOD_CTRL:
                sensor_idx = 2  # right
            step = self.last_state["step"] if self.last_state else 0
            self.sim.fault_injector.trigger_manual_fault(fault_type, sensor_idx, step=step + 1)
        elif event.key == pygame.K_0 and not self.done:
            self.sim.fault_injector.clear_fault()

    def _draw_world(self) -> None:
        panel = pygame.Rect(0, 0, WORLD_PANEL, WINDOW_H)
        pygame.draw.rect(self.screen, PANEL_BG, panel)

        for obs in self.sim.obstacles:
            x0, y0 = world_to_screen(obs.x_min, obs.y_max)
            x1, y1 = world_to_screen(obs.x_max, obs.y_min)
            pygame.draw.rect(self.screen, OBSTACLE_COLOR, pygame.Rect(x0, y0, x1 - x0, y1 - y0))

        gx, gy = world_to_screen(*GOAL_POS)
        pygame.draw.circle(self.screen, GOAL_COLOR, (gx, gy), max(2, int(GOAL_RADIUS * SCALE)), 2)

        if len(self.trail) > 1:
            pts = [world_to_screen(x, y) for x, y in self.trail]
            pygame.draw.lines(self.screen, ROBOT_COLOR, False, pts, 2)

        if self.last_state is not None:
            rx, ry = world_to_screen(self.last_state["x"], self.last_state["y"])
            heading = self.last_state["heading"]
            raw = self.last_state["raw_sensors"]
            observed = self.last_state["observed_sensors"]
            spec = self.last_state["fault_spec"]
            faulty_sensor = spec.sensor_index if spec is not None else None

            for i, angle_deg in enumerate(SENSOR_ANGLES_DEG):
                ang = heading + np.radians(angle_deg)
                dist = observed[i]
                ex = self.last_state["x"] + dist * np.cos(ang)
                ey = self.last_state["y"] + dist * np.sin(ang)
                ex_s, ey_s = world_to_screen(ex, ey)
                if faulty_sensor == i:
                    color = RAY_FAULT_COLOR
                elif dist < 25.0:
                    color = RAY_WARN_COLOR
                else:
                    color = RAY_OK_COLOR
                pygame.draw.line(self.screen, color, (rx, ry), (ex_s, ey_s), 2)
                pygame.draw.circle(self.screen, color, (ex_s, ey_s), 3)

            pygame.draw.circle(self.screen, ROBOT_COLOR, (rx, ry), max(3, int(ROBOT_RADIUS * SCALE)))
            hx = self.last_state["x"] + 4.0 * np.cos(heading)
            hy = self.last_state["y"] + 4.0 * np.sin(heading)
            pygame.draw.line(self.screen, (255, 255, 255), (rx, ry), world_to_screen(hx, hy), 2)

    def _text(self, surface_lines, x, y, font=None, color=TEXT_COLOR, line_h=20):
        font = font or self.font
        for i, line in enumerate(surface_lines):
            img = font.render(line, True, color)
            self.screen.blit(img, (x, y + i * line_h))

    def _draw_sidebar(self) -> None:
        x0 = WORLD_PANEL + 15
        pygame.draw.rect(self.screen, BG, pygame.Rect(WORLD_PANEL, 0, WINDOW_W - WORLD_PANEL, WINDOW_H))

        state = self.last_state
        outcome = state["outcome"] if state else None
        lines = [
            "FAULT-AWARE ROBOT NAV -- DEMO",
            "",
            f"Controller: {self.controller_name}",
            f"Tier:       {self.tier}",
            f"Seed:       {self.seed}",
            f"Speed:      {self.speed}x",
            f"Paused:     {self.paused}",
            "",
        ]
        self._text(lines[:1], x0, 15, font=self.font_big)
        self._text(lines[1:], x0, 50)

        if state is not None:
            fspec = state["fault_spec"]
            fault_line = "Fault: none"
            if fspec is not None:
                fault_line = f"Fault: {fspec.fault_type} on sensor {fspec.sensor_index} (mag={fspec.magnitude:.1f})"
            status_lines = [
                f"Step: {state['step']} / {MAX_STEPS}",
                f"Steering: {state['steer_deg']:+.1f} deg",
                f"Path length: {state['path_length']:.1f}",
                f"Min clearance: {state['min_clearance']:.1f}",
                fault_line,
            ]
            self._text(status_lines, x0, 190)

            sensor_lines = ["Sensors (raw / observed):"]
            names = ["front", "left ", "right"]
            for i, name in enumerate(names):
                sensor_lines.append(
                    f"  {name}: {state['raw_sensors'][i]:5.1f} / {state['observed_sensors'][i]:5.1f}")
            self._text(sensor_lines, x0, 330, font=self.font_small)

            if self.detector is not None:
                det_lines = ["Fault detector (per sensor):"]
                for i, name in enumerate(names):
                    det_lines.append(
                        f"  {name}: {self.detector_labels[i]:<11s} conf={self.detector_confs[i]:.2f}")
                self._text(det_lines, x0, 440, font=self.font_small, color=(160, 220, 255))

            if outcome is not None:
                color = {"success": GOAL_COLOR, "collision": RAY_FAULT_COLOR,
                         "timeout": RAY_WARN_COLOR}.get(outcome, TEXT_COLOR)
                self._text([f"OUTCOME: {outcome.upper()}"], x0, 540, font=self.font_big, color=color)

        controls = [
            "Controls:",
            "SPACE  pause/resume    RIGHT  step once",
            "R      restart episode  N/P    next/prev seed",
            "C      toggle controller T     toggle tier",
            "UP/DOWN adjust speed",
            "1/2/3/4  inject dropout/bias/noise-spike/stale (front)",
            "  +Shift = left sensor, +Ctrl = right sensor",
            "0      clear fault",
        ]
        self._text(controls, x0, 620, font=self.font_small, color=MUTED_TEXT, line_h=17)

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_keys(event)

            if not self.paused or self.step_once:
                n_steps = self.speed if not self.step_once else 1
                for _ in range(n_steps):
                    if not self.done:
                        self._advance()
                self.step_once = False

            self.screen.fill(BG)
            self._draw_world()
            self._draw_sidebar()
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Pygame robot-navigation demo.")
    parser.add_argument("--controller", choices=list(CONTROLLERS.keys()), default="baseline")
    parser.add_argument("--tier", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    app = SimulatorApp(args.controller, args.tier, args.seed)
    app.run()


if __name__ == "__main__":
    main()
