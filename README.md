# RL Template

A modular reinforcement learning training framework built on PyTorch and Gymnasium, providing abstract base classes and a concrete PPO implementation.

## Features

- **Abstract base classes** — `BaseAgent`, `BaseEnv`, `BaseTrain` with template method pattern for easy extension
- **PPO implementation** — Clipped surrogate loss, GAE, linear LR decay, entropy bonus, gradient clipping
- **Pre-allocated buffer** — Zero-allocation rollout storage with bounds checking, supports discrete and continuous actions
- **Frozen configs** — Immutable `PPOConfig` and computed `TrainConfig` dataclasses
- **Auto GPU detection** — Automatic CUDA/CPU device selection
- **107 tests** — Full test suite covering all components

## Project Structure

```
rl_template/
  __init__.py
  agent.py               # BaseAgent — abstract policy/value interface
  env.py                 # BaseEnv — Gymnasium v1 API wrapper
  train.py               # BaseTrain — rollout/update training loop
  common.py              # Buffer — pre-allocated numpy rollout buffer
  config.py              # PPOConfig, TrainConfig, WandbConfig
  errors.py              # EmptyBufferError
  algorithms/
    ppo/
      ppo.py             # PPOTrainer — GAE + clipped surrogate loss
tests/
  test_agent.py          # BaseAgent tests (12)
  test_env.py            # BaseEnv tests (11)
  test_train.py          # BaseTrain tests (6)
  test_common.py         # Buffer tests (21)
  test_config.py         # Config tests (13)
  test_errors.py         # Error tests (7)
  test_ppo.py            # PPOTrainer tests (15)
  test_buffer_integration.py  # Buffer integration tests (6)
  test_continuous_actions.py  # Continuous action tests (16)
```

## Installation

Requires Python 3.11+.

```bash
# With uv (recommended)
uv sync

# With pip
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from rl_template.agent import BaseAgent
from rl_template.env import BaseEnv
from rl_template.train import BaseTrain
from rl_template.common import Buffer
from rl_template.config import TrainConfig, PPOConfig
from rl_template.algorithms.ppo.ppo import PPOTrainer


# 1. Implement your agent
class MyAgent(BaseAgent):
    # Implement forward(), get_distribution()
    ...

# 2. Implement your environment
class MyEnv(BaseEnv):
    # Implement reset(), step(), close()
    ...

# 3. Configure training
config = TrainConfig(model_name="my_agent", model_saved_path="./checkpoints")
ppo_config = PPOConfig(lr=3e-4, gamma=0.99, clip_eps=0.2)

# 4. Wire everything together
agent = MyAgent()
env = MyEnv()
buffer = Buffer(step=config.rollout_steps, state_shape=(4,))
ppo_trainer = PPOTrainer(agent, lr=ppo_config.lr, gamma=ppo_config.gamma)

# 5. Create trainer and run
class MyTrain(BaseTrain):
    def rollout_phase(self, state): super().rollout_phase(state)
    def update_weights(self, step): super().update_weights(step)
    def save_model(self): super().save_model()

trainer = MyTrain(agent, env, buffer, config, ppo_trainer)
state, _ = env.reset()

for step in range(config.num_update):
    trainer.rollout_phase(state)
    trainer.update_weights(step)
    trainer.save_model()
```

## Key Classes

| Class | Description |
|-------|-------------|
| `BaseAgent` | Abstract agent combining ABC and nn.Module. Subclasses implement `forward()` and `get_distribution()`. The `get_action()` template method handles sampling and log-probability computation. |
| `BaseEnv` | Abstract Gymnasium v1 environment wrapper. Subclasses implement `reset()`, `step()`, and `close()`. |
| `BaseTrain` | Abstract training loop orchestrating rollout collection, GAE computation, and PPO updates. |
| `Buffer` | Pre-allocated numpy buffer for rollout data. Supports discrete and continuous actions with automatic dtype handling. |
| `PPOTrainer` | PPO algorithm implementation with GAE, clipped surrogate loss, value loss, entropy bonus, and linear LR decay. |
| `PPOConfig` | Frozen (immutable) PPO hyperparameters. |
| `TrainConfig` | Training configuration with auto-computed `model_path` and `num_update`. |

## PPO Algorithm

The PPO trainer implements:

- **Generalized Advantage Estimation (GAE)** — Blended n-step returns for low-variance advantage estimates
- **Clipped surrogate loss** — Prevents destructive policy updates
- **Value function loss** — MSE loss for the critic network
- **Entropy bonus** — Encourages exploration
- **Linear LR decay** — Learning rate anneals to zero over training
- **Gradient clipping** — Max norm of 1.0 for stability

## Testing

```bash
python -m pytest tests/ -v
```

All 107 tests cover unit tests for each component, integration tests for buffer workflows, and continuous action space validation.

## License

MIT
