"""
ann_imitator.py

Controller E: a small feed-forward network trained to copy the NSGA-II
controller (D). It maps the full observation -- every range reading plus
goal distance and goal angle -- straight to a steering angle and a speed.

Two things about how it is built matter more than the architecture:

Inputs are standardised first. The ranges live in [0, 100] and the goal
angle in [-180, 180], and handing those to a network raw makes the large
columns dominate the early gradients.

Copying the teacher's decisions on the teacher's own routes is not enough.
A network trained that way only ever sees tidy states, so the first small
error puts it somewhere it has no advice for, the next error is bigger, and
it drives into a wall. The earlier version of this did exactly that: it
matched the teacher's steering to an R-squared of 0.75 and still crashed
in every single episode. The fix is in experiments/generate_data.py -- let
the student drive, ask the teacher what it should have done, and add those
states to the training set (Ross, Gordon and Bagnell's DAgger).
"""

from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from controllers.anti_stall import TurnCommit, escape_action
from simulation_core import MAX_SPEED, MAX_STEER_DEG, MIN_SPEED, N_SENSORS
from controllers.goal_bias import (
    CANDIDATE_HEADINGS, clearance_profile, clearance_speed, corridor_clearance,
)
from controllers.fuzzy_handtuned import CRITICAL_AHEAD


class ANNController:
    """Wraps a trained regressor behind the controller interface."""

    def __init__(self, model=None):
        self.model = model
        self._commit = TurnCommit()

    def reset(self) -> None:
        self._commit.reset()

    def predict(self, observation: np.ndarray):
        if self.model is None:
            raise RuntimeError("ANNController has no trained model. "
                                "Load one via ANNController.load(path) first.")
        obs = np.asarray(observation, dtype=np.float64).reshape(1, -1)
        readings = obs.ravel()[:N_SENSORS]

        # Same escape behaviour, same trigger, as the other three. A network
        # that only sees the current readings has no way to remember which
        # way it decided to turn, so without this it deadlocks in exactly the
        # corners its teacher drives out of. Giving it the shared fallback is
        # parity, not a handicap removed -- what is being compared is what
        # each controller does when it still has room to drive.
        ahead = corridor_clearance(readings)
        if ahead < CRITICAL_AHEAD:
            return escape_action(clearance_profile(readings), CANDIDATE_HEADINGS,
                                  ahead, self._commit, MAX_STEER_DEG, clearance_speed)
        self._commit.release()

        out = np.asarray(self.model.predict(obs)).ravel()
        steer, speed = float(out[0]), float(out[1])
        return (float(np.clip(steer, -MAX_STEER_DEG, MAX_STEER_DEG)),
                float(np.clip(speed, MIN_SPEED, MAX_SPEED)))

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save: no trained model set.")
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "ANNController":
        return cls(model=joblib.load(path))


def build_mlp(seed: int = 0) -> Pipeline:
    """Standardiser plus a small MLP, as one estimator.

    Keeping the scaler inside the pipeline means the same transform is
    applied at training time and at inference, with nothing to remember to
    line up by hand later.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=3000,
            early_stopping=True,
            n_iter_no_change=25,
            validation_fraction=0.15,
            random_state=seed,
        )),
    ])
