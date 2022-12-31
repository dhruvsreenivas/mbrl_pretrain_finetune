import torch
from torch import nn, optim
from typing import Generator, Tuple
import numpy as np
import random
import math

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_activation(name: str) -> nn.Module:
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'elu':
        return nn.ELU(inplace=True)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'softmax':
        return nn.Softmax(dim=-1)
    elif name == 'identity':
        return nn.Identity()
    else:
        raise NotImplementedError('Activation not supported.')
    
def get_optimizer(name: str, parameters: Generator, **kwargs) -> optim.Optimizer:
    if name == 'adam':
        return optim.Adam(parameters, **kwargs)
    elif name == 'adamw':
        return optim.AdamW(parameters, **kwargs)
    elif name == 'rmsprop':
        return optim.RMSprop(parameters, **kwargs)
    elif name == 'sgd':
        return optim.SGD(parameters, **kwargs)
    else:
        raise NotImplementedError('Optimizer not supported.')
    
def initialize_weights_tf2(m):
    """Same weight initializations as tf2"""
    if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if type(m) == nn.GRUCell:
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)
        
def agent_weight_init(m):
    """Custom weight init for Conv2d and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        
def stack_dict(x):
    stacked = {}
    if x[0] is None:
        return stacked
    for k in x[0].keys():
        tensors = [d[k] for d in x]
        stacked[k] = torch.stack(tensors)
    return stacked

def flatten_seq(x):
    """(T, B, ...) -> (T * B, ...)"""
    return torch.reshape(x, (-1,) + x.shape[2:])

def unflatten_seq(x, first_dim):
    '''(T * B, ...) -> (T, B, ...)'''
    return torch.reshape(x, (first_dim, -1,) + x.shape[1:])

@torch.no_grad()
def conv_out_dim(init_size: Tuple[int], net: nn.Module):
    h, w = init_size[1], init_size[2]
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            h = (h + 2 * m.padding - m.dilation * (m.kernel_size - 1) - 1) / m.stride + 1
            h = math.floor(h)
            
            w = (w + 2 * m.padding - m.dilation * (m.kernel_size - 1) - 1) / m.stride + 1
            w = math.floor(w)
    return h, w

# ====================== TRAINING UTILS ======================