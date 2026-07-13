"""Abstract base agent interface for reinforcement learning.

Provides BaseAgent, an ABC + nn.Module hybrid that enforces a consistent
policy/value interface for all RL agent implementations.
"""

import numpy as np
import torch.nn as nn
from torch.distributions import Distribution
from torch import Tensor
from abc import ABC, abstractmethod


class BaseAgent(ABC, nn.Module):
    """Abstract base class for all RL agents.

    Subclasses must implement forward() and get_distribution(). The
    get_action() template method composes those two to sample actions,
    compute log probabilities, and return critic values.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state: np.ndarray) -> tuple[Tensor, Tensor]:
        """Compute raw policy logits and value estimate.

        Args:
            state: Environment observation.

        Returns:
            Tuple of (policy_logits, value) tensors.
        """
        pass

    @abstractmethod
    def get_distribution(self, state: np.ndarray) -> tuple[Distribution, Tensor]:
        """Build a distribution over actions for the given state.

        Args:
            state: Environment observation.

        Returns:
            Tuple of (distribution, value) where distribution is a
            torch.distributions.Distribution instance.
        """
        pass

    def get_action(self, state: np.ndarray, action: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample or evaluate an action under the current policy.

        Template method: calls get_distribution(), then samples if no
        action is provided.

        Args:
            state: Environment observation.
            action: Optional pre-selected action. When None, samples from
                    the policy distribution.

        Returns:
            Tuple of (action, log_prob, entropy, value).
        """
        dist, value = self.get_distribution(state)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return (action, log_prob, dist_entropy, value)
