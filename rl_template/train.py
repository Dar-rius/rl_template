import os
import torch
import numpy as np
from env import BaseEnv
from agent import BaseAgent
from algorithms.ppo.ppo import PPOTrainer
from common import Buffer
from config import TrainConfig
from abc import ABC, abstractmethod
from errors import EmptyBufferError


class BaseTrain(ABC):

    def __init__(self, 
                 agent: BaseAgent,
                 env: BaseEnv,
                 buffer: Buffer,
                 train_config: TrainConfig,
                 ppo_trainer: PPOTrainer):
        super().__init__()
        self.agent = agent
        self.env = env
        self.buffer = buffer
        self.train_config = train_config
        self.ppo_trainer = ppo_trainer
        self.last_value = 0.0
        self.cumulative_reward = 0.0
        self.require_buffer_size = 10


    #Rollout phase
    @abstractmethod
    def rollout_phase(self, state: np.ndarray):
        for _ in range(self.train_config.rollout_steps):
            with torch.inference_mode():
                action_t, log_prob, _, value = self.agent.get_action()

            next_state, reward, truncate, done, _ = self.env.step(action_t)
            done_casted = 1 if done else 0

            #Insert data in buffer and variables
            self.buffer.insert(
                state=state,
                action=action_t.item(),
                old_log_prob=log_prob,
                reward=reward,
                value=value,
                dones=done_casted,
            )
            #Update the historic
            self.cumulative_reward += reward

            if done or truncate:
                state = self.env.reset()
            else:
                state = next_state
        #Collect the last critric value
        with torch.inference_mode():
            _, _, _, next_value = self.agent.get_action(state)
        self.last_value = next_value.item()


    @abstractmethod
    def update_weights(self, step:int):
        if self.buffer.size < self.require_buffer_size:
            raise EmptyBufferError(self.buffer.size, self.require_buffer_size)
        rewards_list = self.buffer.rewards
        values_list = self.buffer.values
        dones_list = self.buffer.dones
        #Calcul the GAE
        with torch.inference_mode():
            returns, adv, _ = self.ppo_trainer.compute_gae(rewards_list,
                                                         values_list,
                                                         self.last_value,
                                                         dones_list)
            self.buffer.insert_returns(returns, adv)
        
        #Update the weights
        (loss, policy_loss, value_loss,
         belief_loss, change_loss, entropy) = self.ppo_trainer.update(self.buffer,
                                                                      self.train_config.timestamp,
                                                                      step,
                                                                      self.train_config.batch_size)
        self.buffer.clear()


    @abstractmethod
    def save_model(self):
        if not os.path.exists(self.train_config.model_path):
            os.makedirs(self.train_config.model_path)
        torch.save(self.agent.state_dict(), self.train_config.model_path)
