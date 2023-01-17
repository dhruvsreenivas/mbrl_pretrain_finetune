import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_activation
from memory import Batch

class SingleStepModel(nn.Module):
    '''One single step model. Assume linear MLP.'''
    def __init__(self, cfg):
        super().__init__()
        act = get_activation(cfg.act)
        final_act = get_activation(cfg.final_act)
        self.pred_reward = cfg.pred_reward
        
        layers = []
        dim = cfg.state_dim + cfg.action_dim
        output_dim = cfg.state_dim
        for _ in range(cfg.n_hidden_layers):
            layers.append(nn.Linear(dim, cfg.hidden_dim))
            layers.append(act)
            dim = cfg.hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        self.state_fc = nn.Sequential(
            nn.Linear(dim, output_dim),
            final_act
        )
        if self.pred_reward:
            self.reward_fc = nn.Linear(dim, 1)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        sa = self.trunk(sa)
        
        sp = self.state_fc(sa)
        if self.pred_reward:
            r = self.reward_fc(sa)
            return sp, r
        return sp, None
    
    def loss(self, batch: Batch):
        sp, r = self.forward(batch.states, batch.actions)
        loss = F.mse_loss(sp, batch.next_states)
        metrics = {'state_loss': loss.detach().cpu().item()}
        if r is not None:
            assert r.size() == batch.rewards.size()
            r_loss = F.mse_loss(r, batch.rewards)
            metrics.update(reward_loss=r_loss.detach().cpu().item())
            loss += r_loss
        
        return loss, metrics