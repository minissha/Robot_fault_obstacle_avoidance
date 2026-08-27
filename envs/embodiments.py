"""
Embodiment definitions for Stage 4/5 (second embodiment + shared encoder,
cross-embodiment zero-shot transfer).

IMPORTANT SCOPING NOTE: pybullet_data ships only the Franka Panda arm/
2-finger parallel-jaw gripper — there's no real 3-finger underactuated
gripper URDF available offline in this environment. Rather than block on
an asset we can't fetch (no network access), the second embodiment is
modeled as a *physically distinct gripper variant on the same arm*:
different finger travel range, grasp force, and grasp tolerance, chosen to
approximate how an underactuated 3-finger gripper actually differs from a
parallel-jaw one (wider effective grasp aperture, more forgiving/compliant
contact, looser "is it in-hand" tolerance, and it does NOT need to be
precisely centered over the object before closing).

This keeps the research question intact (does a shared point-cloud
encoder + policy trained across two *physically different* grippers
transfer better under perception noise than an embodiment-specific model)
without depending on an asset you don't have locally. If/when a real
3-finger URDF is available (e.g. Robotiq 3-Finger Adaptive Gripper), swap
`GRIPPER_2F` / `GRIPPER_3F` below for real per-embodiment URDF loading in
`ReachPickPushEnv._build_scene` — the rest of the pipeline (action space,
observation layout, encoder) does not need to change.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbodimentConfig:
    id: str
    name: str
    embodiment_index: int          # for one-hot conditioning, 0-based
    finger_open_target: float      # joint target when "open"
    finger_close_target: float     # joint target when "closed"
    finger_force: float            # motor force used for grasp
    grasp_center_tol: float        # max ee-to-cube dist to count as grasped
    grasp_close_tol: float         # max mean finger position to count as "closed enough"
    action_gripper_deadzone: float = 0.0  # underactuated grippers can be
                                            # sloppier about exact command


GRIPPER_2F = EmbodimentConfig(
    id="2f_parallel_jaw",
    name="2-finger parallel-jaw (Panda default)",
    embodiment_index=0,
    finger_open_target=0.04,
    finger_close_target=0.0,
    finger_force=20.0,
    grasp_center_tol=0.05,
    grasp_close_tol=0.035,
)

GRIPPER_3F = EmbodimentConfig(
    id="3f_underactuated",
    name="3-finger underactuated (SIMULATED: same 2f Panda hardware, "
         "different control params -- not a distinct morphology)",
    embodiment_index=1,
    finger_open_target=0.04,
    finger_close_target=0.008,   # underactuated: doesn't fully close on a
                                  # cube this size, wraps around it instead
    finger_force=12.0,           # softer/compliant closing force
    grasp_center_tol=0.07,       # more forgiving about exact centering
    grasp_close_tol=0.045,       # "closed enough to have wrapped the object"
    action_gripper_deadzone=0.15,
)

ALL_EMBODIMENTS = {cfg.id: cfg for cfg in (GRIPPER_2F, GRIPPER_3F)}
N_EMBODIMENTS = len(ALL_EMBODIMENTS)


def get_embodiment(id_: str) -> EmbodimentConfig:
    if id_ not in ALL_EMBODIMENTS:
        raise ValueError(f"Unknown embodiment '{id_}'. "
                          f"Options: {list(ALL_EMBODIMENTS)}")
    cfg = ALL_EMBODIMENTS[id_]
    if id_ == GRIPPER_3F.id:
        warnings.warn(
            "'3f_underactuated' is the SAME Panda 2-finger parallel-jaw "
            "hardware as '2f_parallel_jaw', run with different finger "
            "travel/force/tolerance parameters -- it is NOT a distinct "
            "gripper morphology (no real 3-finger URDF is loaded). Any "
            "'cross-embodiment' claim built on this pair is really a "
            "cross-*control-parameter* generalization result. See the "
            "module docstring in envs/embodiments.py and the README's "
            "'Asset scoping note' before reporting Stage 4/5 numbers as "
            "morphological cross-embodiment transfer.",
            stacklevel=2,
        )
    return cfg
