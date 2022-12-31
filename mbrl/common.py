import torch
from torch import nn
from torch import distributions as pyd
from utils import *
from mbrl.utils import SquashedNormal
import numpy as np

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        act = get_activation(cfg.act)
        if len(cfg.state_dim) == 3:
            in_channel = cfg.state_dim[0]
            self.trunk = nn.Sequential(
                nn.Conv2d(in_channel, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Flatten()
            )
            out_shape = conv_out_dim(cfg.state_dim, self.trunk)
            self.repr_dim = np.prod(out_shape)
        else:
            state_dim = cfg.state_dim[0]
            self.trunk = nn.Sequential(
                nn.Linear(state_dim, cfg.hidden_size),
                act,
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                act,
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                act,
                nn.Linear(cfg.hidden_size, cfg.hidden_size)
            )
            self.repr_dim = cfg.hidden_size
        
        self.feature_dim = cfg.feature_dim
        self.fc = nn.Linear(self.repr_dim, self.feature_dim)
        self.apply(initialize_weights_tf2)
    
    def forward(self, x: torch.Tensor):
        # x shape: (T, B, dims) because we're supposed to process sequences, or (B, dims) if no sequences
        ndim = x.dim()
        if ndim % 2 == 1:
            T = x.size(0)
            x = flatten_seq(x)
            
        phix = self.trunk(x)
        phix = self.fc(phix)
        
        if ndim % 2 == 1:
            phix = unflatten_seq(phix, T)
            
        return phix
    
class Decoder(nn.Module):
    def __init__(self, feature_dim, repr_dim, cfg):
        super().__init__()
        
        act = get_activation(cfg.act)
        self.fc = nn.Linear(feature_dim, repr_dim)
        
        self.state_ndim = len(cfg.state_dim)
        if self.state_ndim == 3:
            self.unflatten = nn.Unflatten(-1, (repr_dim, 1, 1))
            self.trunk = nn.Sequential(
                nn.ConvTranspose2d(repr_dim, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.ConvTranspose2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.ConvTranspose2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.ConvTranspose2d(cfg.channels, cfg.channels, kernel_size=3, stride=1)
            )
        else:
            state_dim = cfg.state_dim[0]
            self.trunk = nn.Sequential(
                nn.Linear(repr_dim, cfg.hidden_size),
                act,
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                act,
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                act,
                nn.Linear(cfg.hidden_size, state_dim)
            )
        
        self.apply(initialize_weights_tf2)
    
    def forward(self, x: torch.Tensor):
        # x shape: (T, B, feature_dim) or (B, feature_dim)
        ndim = x.dim()
        if ndim % 2 == 1:
            T = x.size(0)
            x = flatten_seq(x)
        
        x = self.fc(x)
        if ndim % 2 == 1:
            x = self.unflatten(x)
        
        mean = self.trunk(x)
        if ndim % 2 == 0:
            mean = unflatten_seq(mean, T)
            
        dist = pyd.Normal(mean, 1.0)
        dist = pyd.Independent(dist, reinterpreted_batch_ndims=self.state_ndim)
        return dist
    
class RewardHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        act = get_activation(cfg.act)
        if len(cfg.state_dim) == 3:
            in_channel = cfg.state_dim[0]
            self.trunk = nn.Sequential(
                nn.Conv2d(in_channel, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Flatten()
            )
            out_shape = conv_out_dim(cfg.state_dim, self.trunk)
            repr_dim = np.prod(out_shape)
        else:
            state_dim = cfg.state_dim[0]
            self.trunk = nn.Sequential(
                nn.Linear(state_dim, cfg.hidden_size),
                act,
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                act,
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                act,
                nn.Linear(cfg.hidden_size, cfg.hidden_size)
            )
            repr_dim = cfg.hidden_size
            
        self.fc = nn.Linear(repr_dim, 1)
    
    def forward(self, state, action):
        pass

class NormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_reset = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_update = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_newval = nn.LayerNorm(hidden_size, eps=1e-3)

    def forward(self, input, state):
        gates_i = self.weight_ih(input)
        gates_h = self.weight_hh(state)
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)

        reset = torch.sigmoid(self.ln_reset(reset_i + reset_h))
        update = torch.sigmoid(self.ln_update(update_i + update_h))
        newval = torch.tanh(self.ln_newval(newval_i + reset * newval_h))
        h = update * newval + (1 - update) * state
        return h

def get_gru(gru_type):
    if gru_type == "gru":
        return nn.GRUCell
    elif gru_type == "gru_layernorm":
        return NormGRUCell
    else:
        raise NotImplementedError("Choose either gru or gru_layernorm")
    
    
# ================== Actor / Critic ==================
class SACActor(nn.Module):
    '''Actor for SAC agent.'''
    def __init__(self, cfg):
        super().__init__()
        assert cfg.min_logstd <= cfg.max_logstd, 'min being larger than max just does not make sense.'
        self.min_logstd = cfg.min_logstd
        self.max_logstd = cfg.max_logstd
        act = get_activation(cfg.act)
        
        if len(cfg.state_dim) == 3:
            in_channel = cfg.state_dim[0]
            self.trunk = nn.Sequential(
                nn.Conv2d(in_channel, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Flatten()
            )
            out_shape = conv_out_dim(cfg.state_dim, self.trunk)
            repr_dim = np.prod(out_shape)
        else:
            state_dim = cfg.state_dim[0]
            self.trunk = nn.Sequential(
                nn.Linear(state_dim, cfg.hidden_size),
                act,
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                act,
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                act,
                nn.Linear(cfg.hidden_size, cfg.hidden_size)
            )
            repr_dim = cfg.hidden_size
        
        self.fc = nn.Linear(repr_dim, 2 * cfg.action_dim)
        self.apply(agent_weight_init)
        
    def forward(self, x):
        phix = self.trunk(x)
        
        mu, logstd = self.fc(phix).chunk(2, dim=-1)
        logstd = torch.tanh(logstd)
        logstd = self.min_logstd + 0.5 * (self.max_logstd - self.min_logstd) * (logstd + 1)
        std = torch.exp(logstd)
        
        dist = SquashedNormal(mu, std)
        return dist
    
class SACCritic(nn.Module):
    '''Critic for SAC agent.'''
    def __init__(self, cfg):
        super().__init__()
        act = get_activation(cfg.act)
        
        if len(cfg.state_dim) == 3:
            in_channel = cfg.state_dim[0]
            self.trunk = nn.Sequential(
                nn.Conv2d(in_channel, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, stride=1),
                act,
                nn.Flatten()
            )
            out_shape = conv_out_dim(cfg.state_dim, self.trunk)
            repr_dim = np.prod(out_shape)
        else:
            state_dim = cfg.state_dim[0]
            self.trunk = nn.Identity()
            repr_dim = state_dim
        
        self.q1 = nn.Sequential(
            nn.Linear(repr_dim + cfg.action_dim, cfg.hidden_size),
            act,
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            act,
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            act,
            nn.Linear(cfg.hidden_size, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(repr_dim + cfg.action_dim, cfg.hidden_size),
            act,
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            act,
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            act,
            nn.Linear(cfg.hidden_size, 1)
        )
        self.apply(agent_weight_init)
        
    def forward(self, s: torch.Tensor, a: torch.Tensor):
        assert s.size(0) == a.size(0), "not the same number of states and actions!"
        
        s = self.trunk(s)
        sa = torch.cat([s, a], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        
        return q1, q2