import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod
from typing import Any


class BaseEnv(ABC):

    def __init__(self):
        self.observation_space: spaces.Space = None
        self.action_space: spaces.Space = None


    @abstractmethod
    def reset(self, seed: int | None = None)-> tuple[np.ndarray, dict[str, Any]]:
        pass


    @abstractmethod
    def step(self, action:np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        pass


    @abstractmethod
    def close(self):
        pass
