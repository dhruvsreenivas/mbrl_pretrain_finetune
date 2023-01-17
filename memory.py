from typing import NamedTuple, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Batch(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    extras: Optional[torch.Tensor] = None
    
class OfflineDataset(Dataset):
    def __init__(self,
                 device,
                 data: Dict[str, np.ndarray]):
        
        super().__init__()
        self.device = torch.device(device)
        
        states = data['observations']
        actions = data['actions']
        rewards = data['rewards']
        next_states = data['next_observations']
        dones = data['terminals']
        
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == next_states.shape[0] == dones.shape[0]
        self.states = torch.from_numpy(states).to(self.device)
        self.actions = torch.from_numpy(actions).to(self.device)
        self.rewards = torch.from_numpy(rewards).to(self.device)
        self.next_states = torch.from_numpy(next_states).to(self.device)
        self.dones = torch.from_numpy(next_states).to(self.device)
        
    def __len__(self):
        return self.states.size(0)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_state = self.next_states[idx]
        done = self.dones[idx]
        return Batch(states=state, actions=action, rewards=reward, next_states=next_state, dones=done)
    
    def sample_seq(self, seq_len: int):
        idx = np.random.randint(low=0, high=len(self))
        
        def closest_start_idx(i):
            less = (self.dones < i).astype(np.int32)
            if np.max(less) == 0:
                return 0
            return np.argmax(np.nonzero(less))
        
        start_idx = closest_start_idx(idx)
        if idx - start_idx > seq_len:
            start_idx = idx - seq_len
        
        s_seq = self.states[start_idx : idx]
        a_seq = self.actions[start_idx : idx]
        r_seq = self.rewards[start_idx : idx]
        ns_seq = self.next_states[start_idx : idx]
        d_seq = self.dones[start_idx : idx]
        
        return Batch(
            states=s_seq,
            actions=a_seq,
            rewards=r_seq,
            next_states=ns_seq,
            dones=d_seq
        )

def dataloader(ds: Dataset, bs: int, nw: int = 1):
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)