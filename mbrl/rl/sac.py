import torch
import torch.nn.functional as F
import numpy as np
from mbrl.common import SACActor, Critic
from mbrl.utils import soft_update_params
from utils import get_optimizer
from memory import Batch

class SAC:
    def __init__(self, cfg):
        # init everything
        self.actor = SACActor(cfg.actor)
        self.critic = Critic(cfg.critic)
        self.target_critic = Critic(cfg.critic)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.log_alpha = torch.tensor(np.log(cfg.init_temp))
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(cfg.action_dim)
        
        # optimizers
        self.actor_opt = get_optimizer(cfg.opt, self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = get_optimizer(cfg.opt, self.critic.parameters(), lr=cfg.lr)
        self.log_alpha_opt = get_optimizer(cfg.opt, [self.log_alpha], lr=cfg.lr)
        
        # hparams
        self.discount = cfg.discount
        self.min_action = cfg.min_action
        self.max_action = cfg.max_action
        self.tau = cfg.tau
        self.actor_update_freq = cfg.actor_update_freq
        self.target_critic_update_freq = cfg.target_update_freq
        self.learn_temp = cfg.learn_temp
        
        self.train()
        self.target_critic.train()
        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        
    def to(self, device: torch.device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        self.target_critic = self.target_critic.to(device)
        self.log_alpha = self.log_alpha.to(device)
        return self
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def act(self, s, sample=False):
        s = torch.from_numpy(s)
        dist = self.actor(s)
        action = dist.sample() if sample else dist.mean
        action = torch.clamp(action, self.min_action, self.max_action)
        return action.cpu().numpy()
    
    def update_critic(self, batch: Batch):
        next_dist = self.actor(batch.next_states)
        next_actions = next_dist.rsample()
        next_lp = next_dist.log_prob(next_actions).sum(-1, keepdim=True)
        nq1, nq2 = self.target_critic(batch.next_states, next_actions)
        nq = torch.min(nq1, nq2) - self.alpha.detach() * next_lp
        nv = batch.rewards + (1.0 - batch.dones) * self.discount * nq
        nv = nv.detach()
        
        # current Q estimates
        q1, q2 = self.critic(batch.states, batch.actions)
        critic_loss = F.mse_loss(q1, nv) + F.mse_loss(q2, nv)
        
        # critic optimization
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        return critic_loss.detach().cpu().item()
    
    def update_actor_and_alpha(self, batch: Batch):
        metrics = {}
        
        dist = self.actor(batch.states)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        
        q1, q2 = self.critic(batch.states, actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha.detach() * log_prob - q).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        metrics['actor_loss'] = actor_loss.detach().cpu().item()
        
        if self.learn_temp:
            self.log_alpha_opt.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_opt.step()
            metrics['alpha_loss'] = alpha_loss.detach().cpu().item()
            
        return metrics
    
    def update(self, batch: Batch, step: int):
        metrics = {}
        critic_loss = self.update_critic(batch)
        metrics['critic_loss'] = critic_loss
        
        if step % self.actor_update_freq == 0:
            actor_alpha_metrics = self.update_actor_and_alpha(batch)
            metrics.update(actor_alpha_metrics)
        
        if step % self.target_critic_update_freq == 0:
            soft_update_params(self.critic, self.target_critic, self.tau)
        
        return metrics