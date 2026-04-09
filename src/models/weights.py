from abc import ABC, abstractmethod

import numpy as np


class Weight(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        pass


class ConstantWeight(Weight):
    def __init__(self, value: float):
        self.value = value

    def __call__(self) -> float:
        return self.value


class RandomWeight(Weight):
    def __init__(self, low: float = 0.0, high: float = 1.0, seed: int | None = None):
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> float:
        return self.rng.uniform(self.low, self.high)
