import numpy as np
import torch.nn as nn
from torch.distributions import Distributions
from torch import Tensor
from abc import ABC, abstractmethod


class BaseAgent(ABC, nn.Module):

    def __init__(self):
        super().__init__()


    @abstractmethod
    def forward(self, state: np.ndarray) -> tuple[Tensor, Tensor]:
        pass


    @abstractmethod
    def get_distribution(self, state: np.ndarray) -> tuple[Distributions, Tensor]:
        pass


    @abstractmethod
    def get_action(self, state: np.ndarray, action:int|None=None) -> tuple[Tensor, Tensor, Tensor, Tensor] :
        dist, value = self.get_distribution(state)
        if action is None: action = dist.sample()
        log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return (action, log_prob, dist_entropy, value)
