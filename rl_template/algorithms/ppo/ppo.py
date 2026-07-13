"""Proximal Policy Optimization (PPO) trainer implementation.

Provides PPOTrainer, which computes GAE advantages and runs the clipped
surrogate loss optimization with linear learning rate decay.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import torch
import torch.nn as nn
from torch import optim
import numpy as np
from ...common import Buffer


class PPOTrainer:
    """PPO trainer handling advantage computation and policy updates.

    Maintains an Adam optimizer and applies linear LR decay over training.

    Args:
        model: Neural network with a get_action(state, action) method.
        lr: Initial learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda (bias-variance tradeoff).
        clip_eps: PPO clipping parameter.
        value_coef: Value loss coefficient.
        ent_coef: Entropy bonus coefficient.
    """

    def __init__(self,
                 model: nn.Module,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_eps: float = 0.2,
                 value_coef: float = 0.5,
                 ent_coef: float = 0.01):
       
        self.model = model
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.ent_coef = ent_coef
        self.mse_loss = nn.MSELoss()

    def compute_gae(self,
                    rewards: np.ndarray,
                    values: np.ndarray,
                    last_value: float,
                    dones: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Works backwards through the trajectory, accumulating TD errors
        with exponentially decaying weights.

        Args:
            rewards: Rewards for each timestep, shape (T,).
            values: Value estimates for each timestep, shape (T,).
            last_value: Bootstrap value for the state after the last step.
            dones: Episode termination flags, shape (T,). 1.0 = done.

        Returns:
            Tuple of (returns, advantages, deltas), each shape (T,).
        """
        gae = 0.0
        # Mask: 0.0 at episode boundaries (no bootstrapping across episodes)
        mask = 1.0 - dones
        next_values = np.concatenate((values[1:], [last_value]), axis=0)
        total_size = rewards.shape[0]
        advantages = np.zeros_like(rewards)

        delta = rewards + self.gamma * next_values * mask - values

        for step in reversed(range(total_size)):
            gae = delta[step] + self.gamma * self.gae_lambda * mask[step] * gae
            advantages[step] = gae

        returns = advantages + values
        return (returns, advantages, delta)

    def lr_decay(self, lr: float, total_steps: int, step: int):
        """Apply linear learning rate decay from lr to 0.

        Args:
            lr: Base learning rate.
            total_steps: Total number of training steps.
            step: Current training step.
        """
        frac = 1.0 - (step / total_steps)
        current_lr = lr * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    def update(self, memory: Buffer, total_steps: int, step: int,
               batch_size: int = 64, epochs: int = 10) -> tuple:
        """Run a PPO update on collected rollout data.

        Normalizes advantages and returns, then runs multiple epochs of
        minibatch SGD with the clipped surrogate loss.

        Args:
            memory: Buffer containing rollout data.
            total_steps: Total training steps (LR decay denominator).
            step: Current training step (LR decay numerator).
            batch_size: Minibatch size.
            epochs: Number of passes over the data.

        Returns:
            Tuple of (total_loss, policy_loss, value_loss, entropy),
            each averaged over all minibatch updates.
        """
        self.lr_decay(self.lr, total_steps, step)

        states, actions, old_log_probs, returns, adv, _, _, _ = memory.get_all()

        advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        dataset_size = actions.size(0)
        num_batch = dataset_size // batch_size

        size_total = int((dataset_size / batch_size) * epochs)
        epoch_losses = torch.zeros((size_total), dtype=torch.float32)
        epoch_pi_losses = torch.zeros((size_total))
        epoch_v_losses = torch.zeros((size_total))
        epoch_entropies = torch.zeros((size_total))
        index_loss = 0

        batch_rollout = torch.arange(0, dataset_size, batch_size)

        for _ in range(epochs):
            shuffle_index = batch_rollout[torch.randperm(num_batch)]

            for start in shuffle_index:
                end = start + batch_size
                idx = torch.arange(start, end)
                if idx.numel() == 0:
                    continue  # Skip empty batches

                _, new_log_probs, dist_entropy, new_values = self.model.get_action(
                    states[idx],
                    actions[idx]
                )

                logratio = new_log_probs - old_log_probs[idx]
                ratio = torch.exp(logratio)

                idx_adv = advantages[idx].flatten()
                surr1 = ratio * idx_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                    1.0 + self.clip_eps) * idx_adv

                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.mse_loss(new_values.flatten(), returns[idx].flatten())

                entropy_loss = dist_entropy.mean()

                loss = policy_loss + \
                        (self.value_coef * value_loss) + \
                        (self.ent_coef * entropy_loss)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_losses[index_loss] = loss.detach()
                epoch_pi_losses[index_loss] = policy_loss.detach()
                epoch_v_losses[index_loss] = value_loss.detach()
                epoch_entropies[index_loss] = entropy_loss.detach()
                index_loss += 1

        # Return average losses over all actual updates ([:index_loss] excludes
        # any unused pre-allocated entries)
        return (epoch_losses[:index_loss].mean().item(),
                epoch_pi_losses[:index_loss].mean().item(),
                epoch_v_losses[:index_loss].mean().item(),
                epoch_entropies[:index_loss].mean().item())
