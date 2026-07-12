import torch
from dataclasses import dataclass, field


# PPO hyper-params
@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-5
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    ent_coef: float = 0.01
    value_coef: float = 0.5
    belief_coef: float = 0.1
    change_coef: float = 0.1


#Training default config
@dataclass
class TrainConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name: str
    model_saved_path: str
    model_path: str = field(init=False)
    timestamp: int = 6_000_000
    batch_size: int = 64
    rollout_steps: int = 2048
    num_update: int = field(init=False)

    def __post_init__(self) -> None:
        self.model_path = f"{self.model_saved_path}/{self.model_name}.pt"
        self.num_update = self.timestamp // self.rollout_steps
        print(f"model is saved in {self.model_path}")
        print(f"{self.device} is used")


#Wandb default config
@dataclass
class WandbConfig:
    name: str
    logs: dict[str, float] = field(default_factory=dict)
