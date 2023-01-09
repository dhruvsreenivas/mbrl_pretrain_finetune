import torch
import torch.nn.functional as F
import numpy as np
from mbrl.common import TD3Actor, Critic
from mbrl.utils import soft_update_params
from utils import get_optimizer
from memory import Batch

class TD3:
    def __init__(self, cfg):
        # init everything
        self.device = torch.device(cfg.device)
        self.actor = TD3Actor(cfg.actor).to(self.device)
        self.target_actor = TD3Actor(cfg.actor).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(cfg.critic).to(self.device)
        self.target_critic = Critic(cfg.critic).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # optimizers
        self.actor_opt = get_optimizer(cfg.opt, self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = get_optimizer(cfg.opt, self.critic.parameters(), lr=cfg.lr)
        
        # hparams
        self.max_action = cfg.max_action
        self.discount = cfg.discount
        self.tau = cfg.tau
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.actor_update_freq = cfg.actor_update_freq
        
        self.train()
        self.target_actor.train()
        self.target_critic.train()
        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        
    def to(self, device: torch.device):
        self.actor = self.actor.to(device)
        self.target_actor = self.target_actor.to(device)
        self.critic = self.critic.to(device)
        self.target_critic = self.target_critic.to(device)
        return self
    
    def act(self, s, sample=False):
        del sample
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).to(self.device)
        
        a = self.actor(s)
        return a
    
    def update_critic(self, batch: Batch):
        next_actions = self.target_actor(batch.next_states)
        noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
        
        # target Q values
        nq1, nq2 = self.target_critic(batch.next_states, next_actions)
        nq = torch.min(nq1, nq2)
        nv = batch.rewards + (1.0 - batch.dones) * self.discount * nq
        nv = nv.detach()
        
        q1, q2 = self.critic(batch.states, batch.actions)
        critic_loss = F.mse_loss(q1, nv) + F.mse_loss(q2, nv)
        
        # optimize
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        return critic_loss.detach().cpu().item()
    
    def update_actor(self, batch: Batch):
        actions = self.actor(batch.states)
        q1 = self.critic.first_q(batch.states, actions)
        actor_loss = -q1.mean()
        
        # optimize
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        return actor_loss.detach().cpu().item()
    
    def update(self, batch: Batch, step: int):
        metrics = {}
        critic_loss = self.update_critic(batch)
        metrics['critic_loss'] = critic_loss
        
        if step % self.actor_update_freq == 0:
            actor_loss = self.update_actor(batch)
            metrics['actor_loss'] = actor_loss
            
            soft_update_params(self.critic, self.target_critic, self.tau)
            soft_update_params(self.actor, self.target_actor, self.tau)
            
        return metrics

        