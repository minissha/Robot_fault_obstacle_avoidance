"""
Stage 1 environment: single embodiment (2-finger parallel-jaw gripper on a
Franka-style arm), ground-truth low-dimensional state, PyBullet physics.

Task: reach a cube on a table, grasp it, and carry/push it into a goal zone
(a marked circle on the table). Episode succeeds if the cube's center stays
inside the goal zone (within `goal_radius`) for `success_hold_steps`
consecutive steps.

This is deliberately the *cheapest and most defensible* first slice of the
project: no pixels, no point clouds, no perception noise. Its only job is to
confirm the task is solvable and the RL/BC pipeline works end to end before
we introduce the real research problem (noisy 3D perception) in Stage 2.

Design choices worth knowing about if you extend this:
- Observation is a flat vector (ground-truth state), NOT an image or point
  cloud. See `_get_obs` for exact layout.
- Action space is 4-dim continuous: delta end-effector (x, y, z) + gripper
  open/close command, applied via inverse-kinematics + a simple PD-style
  position controller. This keeps the action space embodiment-agnostic-ish,
  which matters later for Stage 4 (second embodiment / shared encoder).
- Reward is dense (distance shaping) with sparse bonuses for grasp and
  success, which is standard practice to make PPO converge quickly on a
  laptop CPU within a reasonable number of steps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

from envs.embodiments import EmbodimentConfig, GRIPPER_2F


@dataclass
class EnvConfig:
    render: bool = False
    max_steps: int = 200
    goal_radius: float = 0.06
    success_hold_steps: int = 10
    action_scale: float = 0.03          # max end-effector delta per step (m)
    table_height: float = 0.62
    # NOTE: z-bounds must be in WORLD frame (robot base sits at z=table_height,
    # since the robot is mounted on top of the table). Using table-relative
    # values like (0.0, 0.35) here was the bug that clipped every IK target
    # below the table surface and made the task unsolvable (0% success).
    #
    # workspace_high z was widened from 0.97 -> 1.10: the arm_home reset
    # pose above forward-kinematically resolves to EE z ~= 1.079m, which is
    # *above* the original 0.97 bound. That was a real (if minor)
    # inconsistency -- only the IK *target* for movement was ever clamped
    # to workspace bounds, not this raw reset pose, so nothing caught it
    # before the arm's very first action tried to yank it down ~0.1m to
    # satisfy the (too-tight) clamp. The task's actual working volume never
    # needs more than ~table_height + 0.2 (cube height + lift margin), so
    # this margin only affects where the arm is legally allowed to idle/
    # start, not the reachable task volume.
    workspace_low: tuple = (0.35, -0.25, 0.62)
    workspace_high: tuple = (0.75, 0.25, 1.10)
    seed: int | None = None


class ReachPickPushEnv(gym.Env):
    """Single-embodiment (2-finger gripper) reach/pick/push task."""

    metadata = {"render_modes": ["human", "none"]}

    # Franka Panda joint indices for the 7 arm joints + 2 finger joints
    ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
    FINGER_JOINTS = [9, 10]
    EE_LINK_INDEX = 11

    def __init__(self, config: EnvConfig | None = None,
                 embodiment: EmbodimentConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.embodiment = embodiment or GRIPPER_2F
        self._rng = np.random.default_rng(self.cfg.seed)

        # --- observation space ---
        # [ee_xyz(3), ee_gripper_open(1), cube_xyz(3), cube_quat(4),
        #  goal_xy(2), relative_ee_to_cube(3), relative_cube_to_goal(2),
        #  grasped_flag(1)]  = 19 dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        # action: [dx, dy, dz, gripper_cmd] all in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self._client = None
        self._robot = None
        self._cube = None
        self._goal_marker = None
        self._goal_xy = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._hold_count = 0
        self._grasped = False
        # Explicit grasp attachment. Contact-only grasping (friction between
        # the finger pads and the cube, with no attachment constraint) is
        # known to be unreliable in PyBullet: at 240Hz with a position-
        # controlled gripper, small IK/velocity jitter during transport is
        # enough to break contact and drop the object (this was the root
        # cause of "GRASP LOST" ~step 137 during transport -- the fingers
        # were still commanded closed and still touching the cube in many
        # frames, but contact-only friction just isn't stiff enough to
        # survive sustained accelerated motion). Once a real grasp is
        # detected (fingers closed around the cube, cube centered under the
        # EE), we weld the cube to the EE link with a JOINT_FIXED
        # constraint at the current relative pose, and remove it the moment
        # the gripper is commanded open. This is standard practice for
        # scripted/RL PyBullet manipulation (e.g. the constraint-based
        # grasp used in pybullet's own kuka grasp examples) -- it does not
        # change the grasp *criteria* (still fingers-closed + centered), it
        # just makes the resulting grasp physically stable once achieved,
        # instead of success depending on whether contact friction happens
        # to survive incidental IK jitter.
        self._grasp_constraint_id = None
        self._ever_grasped = False

        self._connect()

    # ------------------------------------------------------------------ #
    # PyBullet setup
    # ------------------------------------------------------------------ #
    # CRITICAL: every PyBullet call in this class must pass
    # `physicsClientId=self._client`. PyBullet supports multiple DIRECT
    # clients in one process (exactly what SB3's DummyVecEnv does with
    # n_envs>1 -- it keeps all N env instances, and their N separate
    # p.connect(p.DIRECT) clients, alive simultaneously in one process,
    # only stepping one at a time), but any call that omits
    # physicsClientId falls back to PyBullet's default client (0), NOT
    # "whichever client this object connected". Previously *no* call
    # anywhere in this file passed it, so with n_envs>1 every environment
    # was silently reading/writing client 0's simulation state instead of
    # its own -- envs 1..N-1 desynced from their own Python-side body/
    # constraint handles the moment client 0 reset or stepped, producing
    # `removeConstraint failed` warnings and physically meaningless
    # rollouts. This is almost certainly why PPO training (n_envs=4)
    # collapsed to 0% success even after the grasp-physics fixes made the
    # single-env (n_envs=1) scripted oracle solve the task at 98%: PPO
    # never saw a coherent, correctly-simulated environment to begin with.
    def _connect(self):
        mode = p.GUI if self.cfg.render else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                   physicsClientId=self._client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self._client)
        if self.cfg.render:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,
                                        physicsClientId=self._client)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.1, cameraYaw=50, cameraPitch=-35,
                cameraTargetPosition=[0.55, 0, 0.4],
                physicsClientId=self._client,
            )

    def _build_scene(self):
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)
        p.loadURDF(
            "table/table.urdf", basePosition=[0.5, 0, 0], globalScaling=1.0,
            physicsClientId=self._client,
        )
        self._robot = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, 0, self.cfg.table_height],
            useFixedBase=True,
            physicsClientId=self._client,
        )
        self._reset_arm_pose()
        self._check_initial_ee_within_workspace()

        cube_xy = self._sample_workspace_xy(margin=0.08)
        cube_pos = [cube_xy[0], cube_xy[1], self.cfg.table_height + 0.05]
        self._cube = p.loadURDF(
            "cube_small.urdf", basePosition=cube_pos, globalScaling=1.0,
            physicsClientId=self._client,
        )
        p.changeDynamics(self._cube, -1, lateralFriction=1.2, mass=0.05,
                          physicsClientId=self._client)

        self._goal_xy = self._sample_workspace_xy(
            margin=0.08, min_dist_from=(cube_xy, 0.15)
        )
        goal_visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=self.cfg.goal_radius, length=0.001,
            rgbaColor=[0.1, 0.8, 0.1, 0.5],
            physicsClientId=self._client,
        )
        self._goal_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_visual,
            basePosition=[self._goal_xy[0], self._goal_xy[1],
                           self.cfg.table_height + 0.001],
            physicsClientId=self._client,
        )

    def _reset_arm_pose(self):
        # 7 arm-joint targets, then 2 finger-joint targets (open = 0.04 each).
        # Previously this zipped an 11-value array (meant for ALL urdf joints,
        # including 2 fixed ones) against only 9 controlled joints, so the
        # fingers silently received 0.0 (closed) instead of 0.04 (open).
        arm_home = [0, -0.4, 0, -2.2, 0, 2.0, 0.78]
        finger_home = [self.embodiment.finger_open_target,
                       self.embodiment.finger_open_target]
        for joint_idx, target in zip(self.ARM_JOINTS, arm_home):
            p.resetJointState(self._robot, joint_idx, target,
                               physicsClientId=self._client)
        for joint_idx, target in zip(self.FINGER_JOINTS, finger_home):
            p.resetJointState(self._robot, joint_idx, target,
                               physicsClientId=self._client)

    def _check_initial_ee_within_workspace(self):
        """Sanity check, run once per scene build: the home-pose EE
        position must actually lie inside workspace_low/high, or every
        episode starts with `_move_ee_to` clamping the very first IK
        target away from where the arm actually is -- silently distorting
        the first several actions. Computed live via FK rather than a
        hardcoded number so it stays correct if arm_home, table_height, or
        the workspace bounds are ever changed independently."""
        ee_pos, _ = self._get_ee_pose()
        low, high = np.array(self.cfg.workspace_low), np.array(self.cfg.workspace_high)
        if np.any(np.array(ee_pos) < low) or np.any(np.array(ee_pos) > high):
            raise RuntimeError(
                f"Initial EE pose {tuple(round(v, 3) for v in ee_pos)} is "
                f"outside workspace bounds {self.cfg.workspace_low} - "
                f"{self.cfg.workspace_high}. arm_home in _reset_arm_pose() "
                f"and EnvConfig.workspace_low/high have drifted out of "
                f"sync -- fix one or the other before training/collecting "
                f"demos, or every episode silently starts with a clamped "
                f"IK target."
            )


    def _sample_workspace_xy(self, margin=0.05, min_dist_from=None):
        low = np.array(self.cfg.workspace_low[:2]) + margin
        high = np.array(self.cfg.workspace_high[:2]) - margin
        for _ in range(50):
            xy = self._rng.uniform(low, high)
            if min_dist_from is None:
                return xy.astype(np.float32)
            ref_xy, min_d = min_dist_from
            if np.linalg.norm(xy - ref_xy) >= min_d:
                return xy.astype(np.float32)
        return xy.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._build_scene()
        self._step_count = 0
        self._hold_count = 0
        self._grasped = False
        # p.resetSimulation(physicsClientId=self._client) inside _build_scene() already destroyed any
        # previous constraint along with the rest of the old scene -- just
        # drop the stale Python-side handle.
        self._grasp_constraint_id = None
        self._ever_grasped = False
        for _ in range(10):
            p.stepSimulation(physicsClientId=self._client)
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        dxyz = action[:3] * self.cfg.action_scale
        gripper_cmd = action[3]  # >0 close, <=0 open

        ee_pos, _ = self._get_ee_pose()
        target_pos = np.array(ee_pos) + dxyz
        target_pos = np.clip(
            target_pos, self.cfg.workspace_low, self.cfg.workspace_high
        )
        self._move_ee_to(target_pos)
        self._set_gripper(gripper_cmd)
        self._update_grasp_constraint(gripper_cmd)

        for _ in range(4):
            p.stepSimulation(physicsClientId=self._client)

        self._step_count += 1
        obs = self._get_obs()
        reward, success, grasped = self._compute_reward(obs)
        self._grasped = grasped

        terminated = success
        truncated = self._step_count >= self.cfg.max_steps
        info = {"success": success, "grasped": grasped}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # GUI already handles rendering when cfg.render=True

    def close(self):
        if self._client is not None:
            p.disconnect(self._client)
            self._client = None

    # ------------------------------------------------------------------ #
    # Low-level robot control
    # ------------------------------------------------------------------ #
    def _get_ee_pose(self):
        state = p.getLinkState(self._robot, self.EE_LINK_INDEX,
                                physicsClientId=self._client)
        return state[0], state[1]  # pos, orientation quat

    def _move_ee_to(self, target_pos):
        down_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        joint_targets = p.calculateInverseKinematics(
            self._robot, self.EE_LINK_INDEX, target_pos, down_orn,
            maxNumIterations=50,
            physicsClientId=self._client,
        )
        for i, joint_idx in enumerate(self.ARM_JOINTS):
            p.setJointMotorControl2(
                self._robot, joint_idx, p.POSITION_CONTROL,
                targetPosition=joint_targets[i], force=150, maxVelocity=1.0,
                physicsClientId=self._client,
            )

    def _set_gripper(self, cmd: float):
        dz = self.embodiment.action_gripper_deadzone
        if cmd > dz:
            target = self.embodiment.finger_close_target
        elif cmd < -dz:
            target = self.embodiment.finger_open_target
        else:
            return  # inside deadzone: hold current target (underactuated slop)
        for joint_idx in self.FINGER_JOINTS:
            p.setJointMotorControl2(
                self._robot, joint_idx, p.POSITION_CONTROL,
                targetPosition=target, force=self.embodiment.finger_force,
                physicsClientId=self._client,
            )

    def _grasp_contact_ok(self):
        """Fingers-closed-ish + cube-centered-under-EE contact criteria.

        Deliberately does NOT require the cube to already be lifted: the
        old version required `cube_pos[2] > table_height + 0.03` *before*
        counting as grasped, but the scripted policy's "grasp" phase only
        transitions to "lift" once grasped -- i.e. it needed the cube
        lifted to detect the grasp that would let it start lifting. That
        circularity only "worked" by accident (fingers squeezing the cube
        sometimes nudges it up a few mm during contact resolution) and is
        exactly the kind of physics-dependent flakiness that also caused
        the mid-transport drops. The lift height is irrelevant to whether a
        grasp has been *formed*; it only matters for whether the grasp is
        being used to lift, which is a separate concern.
        """
        finger_states = [p.getJointState(self._robot, j, physicsClientId=self._client)[0]
                          for j in self.FINGER_JOINTS]
        fingers_closed_ish = (np.mean(finger_states)
                               < self.embodiment.grasp_close_tol)
        cube_pos, _ = p.getBasePositionAndOrientation(self._cube, physicsClientId=self._client)
        ee_pos, _ = self._get_ee_pose()
        close_to_ee = (np.linalg.norm(np.array(cube_pos) - np.array(ee_pos))
                        < self.embodiment.grasp_center_tol)
        return bool(fingers_closed_ish and close_to_ee)

    def _update_grasp_constraint(self, gripper_cmd: float):
        """Attach/detach the cube-to-EE weld constraint based on the
        commanded gripper action, gated by the contact criteria above.
        Called once per env step, before stepSimulation."""
        dz = self.embodiment.action_gripper_deadzone
        if gripper_cmd > dz:
            if self._grasp_constraint_id is None and self._grasp_contact_ok():
                self._attach_grasp()
        elif gripper_cmd < -dz:
            if self._grasp_constraint_id is not None:
                self._release_grasp()
        # inside the deadzone: leave whatever grasp state currently holds
        # alone, same "hold current target" semantics as _set_gripper.

    def _attach_grasp(self):
        ee_pos, ee_orn = self._get_ee_pose()
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self._cube, physicsClientId=self._client)
        ee_inv_pos, ee_inv_orn = p.invertTransform(ee_pos, ee_orn)
        rel_pos, rel_orn = p.multiplyTransforms(
            ee_inv_pos, ee_inv_orn, cube_pos, cube_orn
        )
        self._grasp_constraint_id = p.createConstraint(
            parentBodyUniqueId=self._robot,
            parentLinkIndex=self.EE_LINK_INDEX,
            childBodyUniqueId=self._cube,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos,
            parentFrameOrientation=rel_orn,
            childFramePosition=[0, 0, 0],
            childFrameOrientation=[0, 0, 0],
            physicsClientId=self._client,
        )
        self._ever_grasped = True

    def _release_grasp(self):
        p.removeConstraint(self._grasp_constraint_id, physicsClientId=self._client)
        self._grasp_constraint_id = None

    def _is_grasped(self):
        # Grasped == physically attached. No more inferring "is it still in
        # hand" from friction/contact each frame -- the constraint is the
        # ground truth for grasp state once formed.
        return self._grasp_constraint_id is not None

    # ------------------------------------------------------------------ #
    # Observation / reward
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        ee_pos, _ = self._get_ee_pose()
        finger_states = [p.getJointState(self._robot, j, physicsClientId=self._client)[0]
                          for j in self.FINGER_JOINTS]
        gripper_open = float(np.mean(finger_states) > 0.02)

        cube_pos, cube_quat = p.getBasePositionAndOrientation(self._cube, physicsClientId=self._client)
        grasped = float(self._is_grasped())

        rel_ee_cube = np.array(cube_pos) - np.array(ee_pos)
        rel_cube_goal = np.array(cube_pos[:2]) - self._goal_xy

        obs = np.concatenate([
            np.array(ee_pos, dtype=np.float32),
            np.array([gripper_open], dtype=np.float32),
            np.array(cube_pos, dtype=np.float32),
            np.array(cube_quat, dtype=np.float32),
            self._goal_xy.astype(np.float32),
            rel_ee_cube.astype(np.float32),
            rel_cube_goal.astype(np.float32),
            np.array([grasped], dtype=np.float32),
        ])
        return obs.astype(np.float32)

    def _compute_reward(self, obs: np.ndarray):
        ee_pos = obs[0:3]
        cube_pos = obs[4:7]
        grasped = bool(obs[18])

        dist_ee_cube = np.linalg.norm(ee_pos - cube_pos)
        dist_cube_goal = np.linalg.norm(cube_pos[:2] - self._goal_xy)

        reward = 0.0
        reward += -0.5 * dist_ee_cube          # encourage reaching
        if grasped:
            reward += 1.0                       # grasp bonus (one-off feel via shaping)
            reward += -1.0 * dist_cube_goal      # encourage carrying to goal

        # --- placement / hold logic ---
        # A real pick-place task ends with the gripper OPEN and the object
        # resting in the goal zone -- the policy has to let go to finish.
        # The old condition (`grasped and dist_cube_goal < goal_radius`)
        # made that impossible: the instant the gripper opens to complete
        # the placement, `grasped` flips False and hold_count resets to 0,
        # so a policy that actually releases the cube (rather than idling
        # forever with the gripper clamped shut) could never accumulate
        # `success_hold_steps` and would never be credited with success.
        # Success here instead requires: (a) the cube was genuinely picked
        # up at some point this episode (`_ever_grasped`, rules out just
        # nudging/pushing it into the zone without ever grasping), and
        # (b) the cube is currently settled (low linear speed, so we don't
        # credit it for merely passing through the zone while still being
        # carried or while still falling/sliding) inside the goal radius.
        # This holds equally whether the gripper is still closed on it or
        # has already released it -- releasing no longer invalidates a
        # placement that has already been achieved.
        cube_vel, _ = p.getBaseVelocity(self._cube, physicsClientId=self._client)
        cube_speed = float(np.linalg.norm(cube_vel))
        settled = cube_speed < 0.05  # m/s
        in_goal = dist_cube_goal < self.cfg.goal_radius

        success_condition = self._ever_grasped and in_goal and settled
        if success_condition:
            self._hold_count += 1
        else:
            self._hold_count = 0

        if self._hold_count >= self.cfg.success_hold_steps:
            reward += 10.0
            return reward, True, grasped

        reward -= 0.01  # small time penalty to encourage efficiency
        return reward, False, grasped


def make_env(render: bool = False, seed: int | None = None,
             embodiment: EmbodimentConfig | None = None) -> ReachPickPushEnv:
    return ReachPickPushEnv(EnvConfig(render=render, seed=seed),
                             embodiment=embodiment)


if __name__ == "__main__":
    # Quick manual smoke test: random actions, GUI on.
    env = make_env(render=True, seed=0)
    obs, _ = env.reset()
    for _ in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(1.0 / 60.0)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
