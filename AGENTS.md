# AGENTS.md

## Project

RL training template built on PyTorch + Gymnasium. Provides abstract base classes (`BaseAgent`, `BaseEnv`, `BaseTrain`) and a concrete PPO (Proximal Policy Optimization) implementation. Includes a full test suite (104 tests, all passing).

## Package management

- Managed with **uv** (lockfile: `uv.lock`, Python version: `.python-version` → 3.11)
- Dependencies are pinned in `requirements.txt`, not declared in `pyproject.toml`

## Lint / typecheck

```
ruff check rl_template/       # linter (Pyflakes only, ruff.toml)
mypy rl_template/             # strict: disallow_untyped_defs, warn_unreachable (mypy.ini)
```

No pre-commit hooks, no CI, no formatter config beyond ruff defaults.

## Import style gotcha

- `train.py` uses **relative imports** (`from .env import BaseEnv`) — requires package-level import
- `ppo.py` uses **relative imports** (`from ...common import Buffer`, `from ...config import PPOConfig`) — assumes package-level import
- `algorithms/__init__.py` and `algorithms/ppo/__init__.py` exist (empty) — required for relative imports in `ppo.py` to resolve

## Structure

```
rl_template/
  __init__.py            # empty package marker
  agent.py               # BaseAgent (ABC, nn.Module) — abstract agent interface
  env.py                 # BaseEnv (ABC) — Gymnasium v1 API wrapper interface
  train.py               # BaseTrain (ABC) — rollout/update/save training loop
  common.py              # Buffer — pre-allocated numpy rollout buffer with size property
  config.py              # PPOConfig (frozen), TrainConfig dataclasses
  errors.py              # EmptyBufferError with detailed __str__
  algorithms/
    __init__.py          # package marker (empty)
    ppo/
      __init__.py        # package marker (empty)
      ppo.py             # PPOTrainer — GAE, clipped surrogate loss, linear LR decay
tests/
  __init__.py
  test_common.py         # Buffer unit tests (21 tests)
  test_config.py         # Config dataclass tests (10 tests)
  test_errors.py         # EmptyBufferError tests (7 tests)
  test_ppo.py            # PPOTrainer tests (16 tests)
  test_buffer_integration.py  # Buffer integration tests (6 tests)
  test_agent.py         # BaseAgent abstract interface tests (12 tests)
  test_env.py           # BaseEnv abstract interface tests (11 tests)
  test_train.py         # BaseTrain tests: init, save_model, update_weights (6 tests)
```

## Conventions

- Abstract base classes use `@abstractmethod` with concrete bodies (template method pattern) — subclasses override and optionally call `super()`
- `PPOConfig` is `frozen=True` (immutable); `TrainConfig` uses `__post_init__` for computed fields
- `Buffer` pre-allocates numpy arrays and uses a slice pointer with a `size` property and bounds checking in `insert()`
- GPU detection is automatic in `TrainConfig.device`
- Tests use pytest, import rl_template as a package (`from rl_template.common import Buffer`)

## Testing

```
python -m pytest tests/ -v    # run all 104 tests
```

## Fixes applied

- Created `algorithms/__init__.py` and `algorithms/ppo/__init__.py` for package resolution
- Added `size` property to `Buffer` class (returns current element count)
- Added bounds checking in `Buffer.insert()` — raises `ValueError` when buffer is full
- Fixed `save_model()` to use `os.makedirs(os.path.dirname(...), exist_ok=True)` instead of creating a `.pt` directory
- Fixed PPO `update()` loss tracking — `index_loss` now increments correctly, final mean excludes trailing zeros
- Fixed typo `BaseAgnt` → `BaseAgent` in `agent.py`
- Fixed `ruff.toml` `target-version` from `"py313"` to `"py311"` to match `.python-version`
- Removed `print()` side effects from `TrainConfig.__post_init__`
- Fixed `Buffer.get_all()` actions dtype from `torch.long` to `torch.float32` (continuous actions were silently truncated to integers)
- Fixed `Distributions` → `Distribution` import typo in `agent.py` (pre-existing bug, class is singular in PyTorch)
- Changed `train.py` from bare imports to relative imports for package-level compatibility
- Changed `ppo.py` from `from config import PPOConfig` to `from ...config import PPOConfig` for relative import consistency
- Rewrote all docstrings and comments across source and test files for consistency
- Refactored `PPOTrainer` constructor to accept `PPOConfig` dataclass instead of individual hyperparameter kwargs
- Removed `WandbConfig` dataclass from `config.py`; config module now only exports `PPOConfig` and `TrainConfig`
- Fixed `compute_gae()` to use local `gae` accumulator instead of non-existent `self.ppo_config.gae`
- Updated `PPOConfig` defaults to `lr=3e-5`, `gamma=0.999`, `gae_lambda=0.95`, `clip_eps=0.1`
- Updated all test files to wrap `PPOTrainer` parameters in `PPOConfig(...)`; removed `WandbConfig` tests from `test_config.py`
- Removed unused `import torch` from `tests/test_buffer_integration.py`
- Removed unused `import pytest` from `tests/test_continuous_actions.py`
- Removed unused `from rl_template.common import Buffer` from `tests/test_ppo.py`
