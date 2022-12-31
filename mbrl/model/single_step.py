import torch
import torch.nn as nn
from utils import get_activation

class SingleStepModel(nn.Module):
    '''One single step model. Assume linear MLP.'''
    def __init__(self, cfg):
        super().__init__()
        act = get_activation(cfg.act)
        final_act = get_activation(cfg.final_act)
        
        layers = []
        dim = cfg.state_dim + cfg.action_dim
        output_dim = cfg.state_dim
        for _ in range(cfg.n_hidden_layers):
            layers.append(nn.Linear(dim, cfg.hidden_dim))
            layers.append(act)
            dim = cfg.hidden_dim
        
        layers.append(nn.Linear(dim, output_dim))
        layers.append(final_act)
        self.net = nn.Sequential(*layers)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        s_prime_pred = self.net(sa)
        return s_prime_pred