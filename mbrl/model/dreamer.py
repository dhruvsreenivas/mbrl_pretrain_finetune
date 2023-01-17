import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as pyd
from torch.distributions import kl_divergence
from mbrl.common import *
from utils import *
from memory import Batch

'''Dreamer components + world model.'''

class RSSM(nn.Module):
    '''DreamerV1/V2 RSSM implementation from https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/nets.py -- in PyTorch'''
    def __init__(self, cfg):
        super().__init__()
        
        self.n_models = cfg.n_models
        self.embed_dim = cfg.embed_dim
        self.stoch_dim = cfg.stoc_dim
        self.deter_dim = cfg.deter_dim
        self.hidden_dim = cfg.hidden_dim
        
        self.discrete_dim = cfg.discrete_dim
        self.discrete = (self.discrete_dim > 1)
        
        self.act = get_activation(cfg.act)
        self.std_act = cfg.std_act
        self.min_std = cfg.min_std
        
        # pre gru
        stoch_dim = self.stoch_dim * self.discrete_dim if self.discrete else self.stoch_dim
        self.pre_gru = nn.Sequential(
            nn.Linear(stoch_dim, self.hidden_dim),
            get_norm(cfg.norm_type, self.hidden_dim),
            self.act
        )
        self.gru = get_gru(cfg.gru_type)(self.hidden_dim, self.deter_dim)
        
        # priors and post
        dist_dim = self.stoch_dim * self.discrete_dim if self.discrete else 2 * self.stoch_dim
        self.priors = [
            nn.Sequential(
                nn.Linear(self.deter_dim, self.hidden_dim),
                get_norm(cfg.norm_type, self.hidden_dim),
                self.act,
                nn.Linear(self.hidden_dim, dist_dim)
            )
            for _ in range(self.n_models)
        ]
        self.post = nn.Sequential(
            nn.Linear(self.deter_dim + self.embed_dim, self.hidden_dim),
            get_norm(cfg.norm_type, self.hidden_dim),
            self.act,
            nn.Linear(self.hidden_dim, dist_dim)
        )
        self.apply(initialize_weights_tf2)
        
        # hparams for kl
        self.kl_balancing = cfg.kl.kl_balancing
        self.kl_coef = cfg.kl.kl_coef
        self.free = torch.tensor(cfg.kl.free)
        self.free_avg = cfg.kl.free_avg
        self.post_first = cfg.kl.post_first
        
    @property
    def feature_dim(self):
        if self.discrete:
            return self.deter_dim + self.stoch_dim * self.discrete_dim
        return self.deter_dim + self.stoch_dim
        
    def init_state(self, batch_size: int):
        if self.discrete:
            state = {
                'logits': torch.zeros(batch_size, self.stoch_dim, self.discrete_dim),
                'stoch': torch.zeros(batch_size, self.stoch_dim, self.discrete_dim),
                'deter': torch.zeros(batch_size, self.deter_dim)
            }
        else:
            state = {
                'mean': torch.zeros(batch_size, self.stoch_dim),
                'std': torch.zeros(batch_size, self.stoch_dim),
                'stoch': torch.zeros(batch_size, self.stoch_dim),
                'deter': torch.zeros(batch_size, self.deter_dim)
            }
        return state
    
    def get_dist_and_metrics(self, params, detach=False):
        '''Gets dist from the params.
        
        params: shape (B, stoch_dim, stoch_discrete_dim) if discrete, (B, 2 * stoch_dim) otherwise
        '''
        if detach:
            params = params.detach()
        
        if self.discrete:
            logits = params
            dist = pyd.OneHotCategoricalStraightThrough(logits=logits)
            metrics = {'logits': logits}
        else:
            mean, std = torch.chunk(params, 2, -1)
            if self.std_act == 'sigmoid':
                std = F.sigmoid(std)
            elif self.std_act == 'sigmoid2':
                std = 2 * F.sigmoid(std / 2)
            elif self.std_act == 'softplus':
                std = F.softplus(std)
            else:
                raise NotImplementedError('std activation unimplemented.')
            
            std = std + self.min_std
            dist = pyd.Normal(mean, std)
            metrics = {'mean': mean, 'std': std}
        return pyd.Independent(dist, 1), metrics
    
    def prior_forward(self, prev_state, prev_action, sample=False):
        prev_stoch = prev_state['stoch']
        B = prev_stoch.size(0)
        if self.discrete:
            prev_stoch = torch.reshape(prev_stoch, (B, -1))
        
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self.pre_gru(x)
        
        deter = prev_state['deter']
        deter = self.gru(x, deter)
        
        # get prior dist params
        idx = np.random.randint(0, self.n_models)
        dist_params = self.priors[idx](deter.clone())
        if self.discrete:
            dist_params = torch.reshape(dist_params, (-1, self.stoch_dim, self.discrete_dim))
        
        params = dist_params[idx]
        dist, metrics = self.get_dist_and_metrics(params)
        stoch = dist.rsample() if sample else dist.mean
        state = {
            **metrics,
            'stoch': stoch,
            'deter': deter
        }
        return state
    
    def feature(self, state):
        stoch = state['stoch']
        if self.discrete:
            stoch = torch.reshape(stoch.shape[:-2] + [self.stoch_dim * self.discrete_dim])
        deter = state['deter']
        return torch.cat([stoch, deter], -1)
    
    # losses
    def kl(self, post, prior):
        if self.post_first:
            lhs, rhs = post, prior
            mix = self.kl_coef
        else:
            lhs, rhs = prior, post
            mix = 1.0 - self.kl_coef
        
        lhs_dist, _ = self.get_dist_and_metrics(lhs)
        rhs_dist, _ = self.get_dist_and_metrics(rhs)
        if self.kl_balancing:
            kl = kl_divergence(lhs_dist, rhs_dist)
            kl = torch.maximum(kl, self.free).mean()
        else:
            lhs_dist_sg, _ = self.get_dist_and_metrics(lhs, detach=True)
            rhs_dist_sg, _ = self.get_dist_and_metrics(rhs, detach=True)
            kl_lhs = kl_divergence(lhs_dist, rhs_dist_sg)
            kl_rhs = kl_divergence(lhs_dist_sg, rhs_dist)
            
            if self.free_avg:
                kl_lhs = torch.maximum(kl_lhs.mean(), self.free)
                kl_rhs = torch.maximum(kl_rhs.mean(), self.free)
            else:
                kl_lhs = torch.maximum(kl_lhs, self.free).mean()
                kl_rhs = torch.maximum(kl_rhs, self.free).mean()
                
            kl = mix * kl_lhs + (1.0 - mix) * kl_rhs
        
        return kl
    
    def forward_onestep(self, prev_action, curr_embed, is_first, prev_state=None, sample=True):
        mask = 1.0 - is_first
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        
        B = prev_action.size(1)
        if prev_state is None:
            prev_state = self.init_state(B)
        
        # mask
        prev_state['stoch'] = prev_state['stoch'] * mask
        prev_state['deter'] = prev_state['deter'] * mask
        
        # prior
        prior = self.prior_forward(prev_state, prev_action, sample=sample)
        
        # posterior
        deter = prior['deter']
        x = torch.cat([deter, curr_embed], -1)
        post_params = self.post(x)
        if self.discrete:
            post_params = torch.reshape(post_params, (-1, self.stoch_dim, self.discrete_dim))
            
        post_dist, metrics = self.get_dist_and_metrics(post_params)
        stoch = post_dist.rsample() if sample else post_dist.mean
        state = {
            **metrics,
            'stoch': stoch,
            'deter': deter
        }
        return state, prior
    
    def forward(self, embed_seq, action_seq, is_first_seq, prev_state=None, sample=True):
        priors = []
        posts = []
        features = []
        for embedding, action, is_first in zip(embed_seq, action_seq, is_first_seq):
            post, prior = self.forward_onestep(action, embedding, is_first, prev_state, sample=sample)
            priors.append(prior)
            posts.append(post)
            prev_state = post
            
            feature = self.feature(prev_state)
            features.append(feature)
            
        posts = stack_dict(posts)
        priors = stack_dict(priors)
        features = torch.stack(features)
        return priors, posts, features
    
class WorldModel(nn.Module):
    '''DreamerV1/V2 world model from https://github.com/danijar/dreamerv2/blob/07d906e9c4322c6fc2cd6ed23e247ccd6b7c8c41/dreamerv2/agent.py#L81 in PyTorch.'''
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder = Encoder(cfg.encoder)
        self.rssm = RSSM(cfg.rssm)
        self.decoder = Decoder(self.rssm.feature_dim, cfg.state_dim, cfg.decoder)
        
        self.pred_reward = cfg.pred_reward
        if self.pred_reward:
            self.reward_head = RewardHead(self.rssm.feature_dim, cfg.reward)
            
        self.reward_coef = cfg.reward_coef
        self.kl_coef = cfg.kl_coef
        self.state_ndim = self.decoder.state_ndim

        self.apply(initialize_weights_tf2)
        
    def forward(self, state_seq, action_seq, is_first_seq):
        states = flatten_seq(state_seq)
        embeds = self.encoder(states)
        embed_seq = unflatten_seq(embeds) # (T, B, embed_dim)
        
        posts, priors, features = self.rssm(embed_seq, action_seq, is_first_seq)
        return posts, priors, features
        
    def reward_loss(self, features, rewards):
        assert self.pred_reward, "Can't predict rewards when no reward head is present"
        reward_mean = self.reward_head(features)
        dist = pyd.Normal(reward_mean, 1)
        lp = dist.log_prob(rewards)
        return -lp.mean()
    
    def reconstruct_loss(self, features, states):
        rec_mean = self.decoder(features)
        dist = pyd.Normal(rec_mean, 1.0)
        dist = pyd.Independent(dist, self.state_ndim)
        lp = dist.log_prob(states)
        return -lp.mean()
    
    def loss(self, batch: Batch):
        posts, priors, features = self.forward(batch.states, batch.actions, batch.extras)
        
        rec_loss = self.reconstruct_loss(features, batch.states)
        kl_loss = self.rssm.kl(posts, priors)
        reward_loss = self.reward_loss(features, batch.rewards)
        
        metrics = {
            'rec_loss': rec_loss.detach().cpu().item(),
            'kl': kl_loss.detach().cpu().item(),
            'reward_loss': reward_loss.detach().cpu().item()
        }
        
        return rec_loss + self.kl_coef * kl_loss + self.reward_coef * reward_loss, metrics
        