"""
Hand-scripted reach -> grasp -> transport -> release controller.

Uses ground-truth state only (reads directly from the obs vector produced by
ReachPickPushEnv). This is intentionally simple and not meant to be a great
policy -- it exists to (a) sanity check the environment/reward are correct,
and (b) generate clean demonstrations for behavior cloning in Stage 1.

Obs layout (see envs/reach_pick_push_env.py::_get_obs):
  [0:3]   ee_xyz
  [3]     gripper_open (1.0 open / 0.0 closed-ish)
  [4:7]   cube_xyz
  [7:11]  cube_quat
  [11:13] goal_xy
  [13:16] rel_ee_to_cube (cube - ee)
  [16:18] rel_cube_to_goal (cube_xy - goal_xy)
  [18]    grasped_flag
"""
from __future__ import annotations

import numpy as np


class ScriptedPickPushPolicy:
    APPROACH_HEIGHT = 0.10   # hover this far above cube before descending
    GRASP_HEIGHT_OFFSET = 0.0
    # Must safely exceed EnvConfig.success_hold_steps (default 10): the env
    # only credits success once the cube has been settled in the goal
    # radius for that many consecutive steps, and (as of the grasp/hold
    # fix) the cube only counts as "settled" once it has stopped moving.
    # We hold noticeably longer than the env's threshold so small execution
    # noise near the goal boundary doesn't cost us a full re-accumulation
    # of hold_count, then explicitly release only after that.
    HOLD_STEPS = 20

    def __init__(self):
        self._phase = "approach"  # approach -> descend -> grasp -> lift -> transport -> hold -> release
        self._hold_counter = 0
        self._lift_target_z = None

    def reset(self):
        self._phase = "approach"
        self._hold_counter = 0
        self._lift_target_z = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        ee_pos = obs[0:3]
        cube_pos = obs[4:7]
        goal_xy = obs[11:13]
        grasped = bool(obs[18])

        target_xy_above_cube = cube_pos[:2]
        dx_dy_to_cube = target_xy_above_cube - ee_pos[:2]
        horiz_dist = np.linalg.norm(dx_dy_to_cube)

        if self._phase == "approach":
            target = np.array([cube_pos[0], cube_pos[1],
                                cube_pos[2] + self.APPROACH_HEIGHT])
            delta = target - ee_pos
            if horiz_dist < 0.015 and abs(ee_pos[2] - target[2]) < 0.02:
                self._phase = "descend"
            return self._to_action(delta, gripper_close=False)

        if self._phase == "descend":
            target = np.array([cube_pos[0], cube_pos[1],
                                cube_pos[2] + self.GRASP_HEIGHT_OFFSET + 0.01])
            delta = target - ee_pos
            if np.linalg.norm(delta) < 0.015:
                self._phase = "grasp"
            return self._to_action(delta, gripper_close=False)

        if self._phase == "grasp":
            if grasped:
                # Latch the lift target ONCE, from the cube's position at
                # the moment the grasp forms. The old version recomputed
                # `target[2] = cube_pos[2] + 0.15` from the *live* cube
                # position every step of "lift" -- harmless when grasping
                # was flaky (the cube didn't perfectly track the EE), but
                # with a physically solid grasp (see env's grasp
                # constraint) the cube now tracks the EE exactly, so that
                # live target chases the EE upward forever and "lift" never
                # terminates. Latching it to a fixed world-frame height at
                # grasp time fixes that.
                self._lift_target_z = cube_pos[2] + 0.15
                self._phase = "lift"
            return self._to_action(np.zeros(3), gripper_close=True)

        if self._phase == "lift":
            target = ee_pos.copy()
            target[2] = self._lift_target_z
            delta = target - ee_pos
            if delta[2] < 0.02:
                self._phase = "transport"
            return self._to_action(delta, gripper_close=True)

        if self._phase == "transport":
            target = np.array([goal_xy[0], goal_xy[1], ee_pos[2]])
            delta = target - ee_pos
            if np.linalg.norm(delta[:2]) < 0.02:
                self._phase = "hold"
                self._hold_counter = 0
            return self._to_action(delta, gripper_close=True)

        if self._phase == "hold":
            # Stay put, gripper still closed, and let the cube settle
            # (zero velocity) in the goal zone for long enough that the
            # env's own hold-count threshold is guaranteed to be cleared
            # before we let go. Releasing before this point is exactly the
            # bug that used to make "reaches the goal but never registers
            # success" possible: we must not open the gripper on first
            # contact with the goal radius.
            self._hold_counter += 1
            if self._hold_counter >= self.HOLD_STEPS:
                self._phase = "release"
            return self._to_action(np.zeros(3), gripper_close=True)

        if self._phase == "release":
            return self._to_action(np.zeros(3), gripper_close=False)

        return np.zeros(4, dtype=np.float32)

    @staticmethod
    def _to_action(delta_xyz: np.ndarray, gripper_close: bool,
                    action_scale: float = 0.03) -> np.ndarray:
        capped = np.clip(delta_xyz / action_scale, -1.0, 1.0)
        gripper_cmd = 1.0 if gripper_close else -1.0
        return np.concatenate([capped, [gripper_cmd]]).astype(np.float32)
