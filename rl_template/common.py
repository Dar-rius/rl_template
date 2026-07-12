import torch
import numpy as np

class Buffer:
    """
    1 -> State
    2 -> Action
    3 -> Old Log Probs 
    4 -> Returns
    5 -> Advantage
    6 -> Reward
    7 -> Value
    8 -> Dones
    """
    def __init__(self, step:int, state_shape:tuple, action_shape:tuple=()):
        self.step = step
        self.slice: int = 0
        self.states = np.zeros((step, state_shape), dtype=np.float32)
        self.actions = np.zeros((step, action_shape), dtype=np.float32)
        self.old_log_probs = np.zeros(self.step, dtype=np.float32)
        self.returns = np.zeros(self.step, dtype=np.float32)
        self.adv = np.zeros(self.step, dtype=np.float32)
        self.rewards = np.zeros(self.step, dtype=np.float32)
        self.values = np.zeros(self.step, dtype=np.float32)
        self.dones = np.zeros(self.step, dtype=np.float32)

    def insert(self, 
               state:np.ndarray,
               action:int,
               old_log_prob:float,
               reward:float,
               value:float,
               dones:int,
               ):
        self.states[self.slice] = state
        self.actions[self.slice] = action
        self.old_log_probs[self.slice] = old_log_prob
        self.rewards[self.slice] = reward
        self.values[self.slice] = value
        self.dones[self.slice] = dones
        self.slice += 1

    def insert_returns(self, returns:np.ndarray, adv:np.ndarray):
        self.returns[:] = returns
        self.adv[:] = adv
    
    # sampling data
    def get_all(self) -> tuple:
        return (torch.tensor(self.states, dtype=torch.float32),
                torch.tensor(self.actions, dtype=torch.long),
                torch.tensor(self.old_log_probs, dtype=torch.float32),
                torch.tensor(self.returns, dtype=torch.float32),
                torch.tensor(self.adv, dtype=torch.float32),
                torch.tensor(self.rewards, dtype=torch.float32),
                torch.tensor(self.values, dtype=torch.float32),
                torch.tensor(self.dones, dtype=torch.long))

    # Delete all data
    def clear(self):
        self.slice = 0
