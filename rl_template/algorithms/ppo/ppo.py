import torch
import torch.nn as nn
from torch import optim
import numpy as np
from ...common import Buffer


# PPO Implementation
class PPOTrainer:
    def __init__(self,
                 model:nn.Module,
                 lr:float=3e-4,
                 gamma:float=0.99,
                 gae_lambda:float=0.95,
                 clip_eps:float=0.2,
                 value_coef:float=0.5,
                 ent_coef:float=0.01,
                ):
        self.model = model
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        #PPO Hyperparams
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        # Total Loss Coefficients
        self.value_coef = value_coef
        self.ent_coef = ent_coef
        self.mse_loss = nn.MSELoss()

    
    def compute_gae(self, 
                    rewards:np.ndarray,
                    values:np.ndarray,
                    last_value:float,
                    dones:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gae = 0.0
        mask = 1.0 - dones
        next_values = np.concatenate((values[1:], [last_value]), axis=0)
        total_size = rewards.shape[0]
        advantages = np.zeros_like(rewards)
        delta = rewards + self.gamma * next_values * mask - values
        for step in reversed(range(total_size)):
            gae = delta[step] + self.gamma * self.gae_lambda * mask[step] * gae
            advantages[step] =  gae
        returns = advantages + values
        return (returns, advantages, delta)

    def lr_decay(self, lr:float, total_steps:int, step:int):
        frac = 1.0 - (step / total_steps)
        current_lr = lr * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr


    # Calcul alla loss for all auxillary task and updates weights
    def update(self, memory:Buffer, total_steps:int, step:int, batch_size:int=64, epochs:int=10) -> tuple:
        self.lr_decay(self.lr, total_steps, step)
        states, actions, old_log_probs, returns, adv, _, _, _ = memory.get_all()
        # Normalize the advantages and returns
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
                if idx.numel() == 0: continue

                _, new_log_probs, dist_entropy, new_values, _,  _ = self.model.get_action(
                        states[idx],
                        actions[idx]
                        )

                # Calcul Ratio (new Policy / old Policy)
                logratio = new_log_probs - old_log_probs[idx]
                ratio = torch.exp(logratio)

                # Calcul the PPO Loss
                idx_adv = advantages[idx].flatten()
                surr1 = ratio * idx_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * idx_adv
                # Calcul the policy loss (Actor)
                policy_loss = -torch.min(surr1, surr2).mean()
                # Calcul the value loss (Critic)
                value_loss = self.mse_loss(new_values.flatten(), returns[idx].flatten())
                entropy_loss = dist_entropy.mean()

                # Update weights
                loss = policy_loss + \
                        (self.value_coef * value_loss) + \
                        (self.ent_coef * entropy_loss)
                # Backpropagation
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                #Store all data about loss
                epoch_losses[index_loss] = loss
                epoch_pi_losses[index_loss] = policy_loss
                epoch_v_losses[index_loss] = value_loss
                epoch_entropies[index_loss] = entropy_loss

        return (epoch_losses.mean().item(),
                epoch_pi_losses.mean().item(),
                epoch_v_losses.mean().item(),
                epoch_entropies.mean().item())
