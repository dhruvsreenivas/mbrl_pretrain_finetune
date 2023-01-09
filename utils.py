import torch
from torch import nn, optim
from typing import Iterator, Tuple, List, Any
import numpy as np
import gym
import random
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

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
    
def get_optimizer(name: str, parameters: Iterator, **kwargs) -> optim.Optimizer:
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
    
def get_norm(norm_type: str, *args):
    if norm_type == 'none':
        return nn.Identity()
    elif norm_type == 'layer':
        return nn.LayerNorm(*args)
    else:
        raise NotImplementedError('norm not implemented haha')
    
def initialize_weights_tf2(m):
    """Same weight initializations as tf2"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if isinstance(m, nn.GRUCell):
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
def conv_out_size(init_size: Tuple[int], net: nn.Module):
    '''Standard out dim of a convnet (not convtranspose!).'''
    x = torch.randn(*init_size)
    for m in net.children():
        if not isinstance(m, nn.Flatten):
            x = m(x) # shouldn't mess up anything here because nn.Flatten only happens at the end
    return x.size()

# ====================== TRAINING UTILS ======================

class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until

class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False
    
class AverageMeter:
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)
    
# ====================== DATA COLLECTION UTILS ======================
def train_agent(env: gym.Env, steps=3000000, save_interval=10000, save_rb=False) -> List[Any]:
    '''Trains a SAC agent in the env, saving checkpoints as needed.'''
    print(env.name)
    model = SAC(
        policy='MlpPolicy',
        env=env
    )
    chkptr = CheckpointCallback(
        save_freq=save_interval,
        save_path='./online_agents',
        name_prefix=f'{env.name}_sac_online',
        save_replay_buffer=save_rb
    )
    model.learn(steps, callback=chkptr)
    
if __name__ == '__main__':
    # test agent training
    import metaworld
    mt10 = metaworld.MT50()
    tasks_of_interest = ['door-open-v2', 'door-close-v2', 'drawer-open-v2', 'drawer-close-v2']
    
    train_envs = []
    for name in tasks_of_interest:
        print(name)
        env = mt10.train_classes[name]()
        tasks = [task for task in mt10.train_tasks if task.env_name == name]
        count = 0
        for task in tasks:
            env.set_task(task)
            env.name = f'{name}_task{count}'
            train_envs.append(env)
            count += 1
    
    print(len(train_envs))
    
    
    