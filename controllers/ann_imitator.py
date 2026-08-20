"""
ann_imitator.py

Controller E ("C" in the blueprint): small feed-forward ANN regression
controller trained by imitation on the NSGA-II-tuned FLC (Controller D).

Maps [front, left, right] sensor readings directly to a steering angle
in degrees, via a scikit-learn MLPRegressor.
"""

from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor


class ANNController:
    """Wraps a trained (or untrained) MLPRegressor behind the controller interface."""

    def __init__(self, model: Optional[MLPRegressor] = None):
        self.model = model

    def reset(self) -> None:
        return None

    def predict(self, sensor_readings: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("ANNController has no trained model. "
                                "Load one via ANNController.load(path) first.")
        x = np.asarray(sensor_readings, dtype=np.float64).reshape(1, -1)
        steer = float(self.model.predict(x)[0])
        return steer

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save: no trained model set.")
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "ANNController":
        model = joblib.load(path)
        return cls(model=model)


def build_mlp(seed: int = 0) -> MLPRegressor:
    """Construct a small MLP regressor with early stopping, as per the blueprint."""
    return MLPRegressor(
        hidden_layer_sizes=(20,),
        activation="relu",
        solver="adam",
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=seed,
    )
