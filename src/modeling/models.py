from __future__ import annotations

import numpy as np
from dowhy.gcm.ml import PredictionModel


class SumModel(PredictionModel):
    def __init__(self, limits: tuple[int, int]) -> None:
        self.a, self.b = limits

    def fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert X.shape[1] == 2
        return np.atleast_2d(
            np.clip(
                X[:, 0] + X[:, 1],
                self.a, self.b
            )
        ).T

    def clone(self) -> SumModel:
        return SumModel((self.a, self.b))
