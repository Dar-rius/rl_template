"""Unit tests for BaseTrain (rl_template.train)."""

import os
import numpy as np
import torch
import torch.nn as nn
import pytest

from rl_template.agent import BaseAgent
from rl_template.env import BaseEnv
from rl_template.train import BaseTrain
from rl_template.common import Buffer
from rl_template.config import TrainConfig, PPOConfig
from rl_template.algorithms.ppo.ppo import PPOTrainer
from rl_template.errors import EmptyBufferError


class MockAgent(BaseAgent):
    """Minimal agent for testing BaseTrain."""

    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.linear = nn.Linear(obs_dim, act_dim)
        self.val = nn.Linear(obs_dim, 1)

    def forward(self, state, **kwargs):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        return self.linear(state_t), self.val(state_t)

    def get_distribution(self, state, **kwargs):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value.squeeze(-1)

class MockEnv(BaseEnv):
    """Minimal environment for testing BaseTrain."""

    def __init__(self, obs_dim=4):
        self.obs_dim = obs_dim
        self._step_count = 0

    def reset(self, seed=None):
        self._step_count = 0
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = np.ones(self.obs_dim, dtype=np.float32)
        done = self._step_count >= 10
        return obs, 1.0, done, False, {}

    def close(self):
        pass

class TestBaseTrainInit:
    """Verify BaseTrain stores all dependencies."""

    def test_stores_attributes(self, tmp_path):
        obs_dim, act_dim = 4, 2
        agent = MockAgent(obs_dim, act_dim)
        env = MockEnv(obs_dim)
        buf = Buffer(step=10, state_shape=(obs_dim,))
        cfg = TrainConfig(model_name="test", model_saved_path=str(tmp_path))
        ppo = PPOTrainer(agent, PPOConfig())

        #Check type
        trainer = BaseTrain(agent, env, buf, cfg, ppo)
        assert trainer.agent is agent
        assert trainer.env is env
        assert trainer.buffer is buf
        assert trainer.train_config is cfg
        assert trainer.ppo_trainer is ppo

    def test_default_last_value(self, tmp_path):
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, state_shape=(4,))
        cfg = TrainConfig(model_name="test", model_saved_path=str(tmp_path))
        ppo = PPOTrainer(agent, PPOConfig())

        trainer = BaseTrain(agent, env, buf, cfg, ppo)
        assert trainer.last_value == 0.0

    def test_default_cumulative_reward(self, tmp_path):
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, state_shape=(4,))
        cfg = TrainConfig(model_name="test", model_saved_path=str(tmp_path))
        ppo = PPOTrainer(agent, PPOConfig())

        trainer = BaseTrain(agent, env, buf, cfg, ppo)
        assert trainer.cumulative_reward == 0.0


class TestBaseTrainUpdateWeights:
    """Verify update_weights() raises error on empty buffer."""

    def test_raises_empty_buffer_error(self, tmp_path):
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, state_shape=(4,))
        cfg = TrainConfig(model_name="test", model_saved_path=str(tmp_path))
        ppo = PPOTrainer(agent, PPOConfig())

        trainer = BaseTrain(agent, env, buf, cfg, ppo)
        with pytest.raises(EmptyBufferError):
            trainer.update_weights(step=0)


class TestBaseTrainSaveModel:
    """Verify save_model() writes weights to disk."""

    def test_creates_file(self, tmp_path):
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, state_shape=(4,))
        save_dir = str(tmp_path / "models")
        cfg = TrainConfig(model_name="test", model_saved_path=save_dir)
        ppo = PPOTrainer(agent, PPOConfig())

        trainer = BaseTrain(agent, env, buf, cfg, ppo)
        trainer.save_model()
        assert os.path.exists(cfg.model_path)

    def test_saved_weights_are_loadable(self, tmp_path):
        agent = MockAgent()
        env = MockEnv()
        buf = Buffer(step=10, state_shape=(4,))
        save_dir = str(tmp_path / "models")
        cfg = TrainConfig(model_name="test", model_saved_path=save_dir)
        ppo = PPOTrainer(agent, PPOConfig())

        trainer = BaseTrain(agent, env, buf, cfg, ppo)
        trainer.save_model()

        new_agent = MockAgent()
        state_dict = torch.load(cfg.model_path, weights_only=True)
        new_agent.load_state_dict(state_dict)
        for p1, p2 in zip(agent.parameters(), new_agent.parameters()):
            assert torch.allclose(p1, p2)
