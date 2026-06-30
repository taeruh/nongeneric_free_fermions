from abc import ABC, abstractmethod

import numpy as np


class Weight(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.float64:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class ConstantWeight(Weight):
    def __init__(self, value: np.float64 | float):
        self.value = np.float64(value)

    def __call__(self) -> np.float64:
        return self.value

    def __repr__(self) -> str:
        return f"ConstantWeight({self.value})"


class RandomWeight(Weight):
    def __init__(
        self,
        low: np.float64 | float = np.float64(0.0),
        high: np.float64 | float = np.float64(1.0),
        seed: int | None = None,
    ):
        self.seed = seed
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> np.float64:
        return np.float64(self.rng.uniform(self.low, self.high))

    def __repr__(self) -> str:
        return f"RandomWeight({self.low},{self.high},{self.seed})"
