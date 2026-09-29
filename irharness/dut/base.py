from __future__ import annotations

import abc

import numpy as np


class DUT(abc.ABC):
    def __init__(self, spec: str):
        self.spec = spec

    @abc.abstractmethod
    def run(self, x: np.ndarray, fs: int) -> np.ndarray:
        """Return the response to x (float64, same fs). May be longer than x."""

    def describe(self) -> dict:
        return {"spec": self.spec}
