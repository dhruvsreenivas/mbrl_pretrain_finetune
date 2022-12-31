import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import kl_divergence
from mbrl.common import *
from utils import *
import numpy as np

'''Dreamer components + world model.'''

class RSSMCell(nn.Module):
    def __init__(self, embed_dim, cfg):
        super().__init__()
        
        self.deter_dim = cfg.deter_dim
        self.stoc_dim = cfg.stoc_dim
        self.stoc_discrete_dim = cfg.stoc_discrete_dim
        self.n_models = cfg.n_ensemble
        self.discrete = cfg.discrete
        self.feature_dim = self.deter_dim + (
            self.stoc_dim * (self.stoc_discrete_dim if self.discrete else 1)
        )

        self.act = get_activation(cfg.act)

        # Create Recurrent Stack
        self.gru = get_gru(cfg.gru_type)(cfg.hidden_dim, self.deter_dim)
        self.pre_gru = nn.Linear(
            self.stoc_dim * (self.stoc_discrete_dim if self.discrete else 1)
            + cfg.n_actions,
            cfg.hidden_dim,
        )

        # Create Ensemble of MLPs here
        dist_dim = self.stoc_dim * (self.stoc_discrete_dim if self.discrete else 2)
        # only 1 prior due to online finetuning -- don't need ensemble necessarily
        self.prior_mlp = nn.Sequential(
            nn.Linear(self.deter_dim, cfg.hidden_dim),
            self.act,
            # 32 * 32
            nn.Linear(cfg.hidden_dim, dist_dim),
        )
        self.post_mlp = nn.Sequential(
            nn.Linear(self.deter_dim + embed_dim, cfg.hidden_dim),
            self.act,
            # 32 * 32
            nn.Linear(cfg.hidden_dim, dist_dim),
        )

        # Initialize Weights
        self.apply(initialize_weights_tf2)

    def init_state(self, batch_size):
        """
        Getting the initial RNN state
        """
        device = next(self.gru.parameters()).device
        if self.discrete:
            state = {
                "deter": torch.zeros((batch_size, self.deter_dim), device=device),
                "stoc": torch.zeros(
                    (batch_size, self.stoc_dim * self.stoc_discrete_dim), device=device
                ),
            }
        else:
            state = {
                "deter": torch.zeros((batch_size, self.deter_dim), device=device),
                "stoc": torch.zeros((batch_size, self.stoc_dim), device=device),
            }
        return state

    def prior_forward(self, action, mask, hidden_state, sample=True):
        deter_state, prev_latent = hidden_state["deter"], hidden_state["stoc"]

        # Reset Masks
        deter_state *= mask
        prev_latent *= mask

        x = torch.cat([prev_latent, action], dim=-1)
        x = self.act(self.pre_gru(x))
        # Note the hidden state encodes deterministic dynamics
        deter_state = self.gru(x, deter_state)
        # Take the k'th model for next imagined latent
        prior = self.prior_mlp(deter_state)
        prior_dist, prior_stats = self.zdist(prior)
        if sample:
            sample_latent = prior_dist.rsample().reshape(action.size(0), -1)
        else:
            sample_latent = prior_dist.mean.reshape(action.size(0), -1)
        return prior_stats, {"deter": deter_state, "stoc": sample_latent}

    def zdist(self, post_prior):
        # Either returns One hot Categorical or Multivariate Normal Diagonal
        if self.discrete:
            logits = post_prior.reshape(
                post_prior.shape[:-1] + (self.stoc_dim, self.stoc_discrete_dim)
            )
            stats = {"logits": logits}
        else:
            mean, std = post_prior.chunk(2, -1)
            std = F.softplus(std) + 0.1
            stats = {"mean": mean, "std": std}
        return self.get_dist(stats), stats

    def get_dist(self, stats):
        if self.discrete:
            dist = torch.distributions.OneHotCategoricalStraightThrough(
                logits=stats["logits"]
            )
        else:
            dist = torch.distributions.normal.Normal(stats["mean"], stats["std"])
        return torch.distributions.Independent(dist, 1)

    def get_feature(self, state):
        return torch.cat([state["deter"], state["stoc"]], -1)

    def forward(
        self, embedding, action, mask, hidden_state, model_idx=None, sample=True
    ):
        deter_state, prev_latent = hidden_state["deter"], hidden_state["stoc"]

        # Reset Masks
        deter_state *= mask
        prev_latent *= mask

        x = torch.cat([prev_latent, action], dim=-1)
        x = self.act(self.pre_gru(x))
        # Note the hidden state encodes deterministic dynamics
        deter_state = self.gru(x, deter_state)

        # Prior
        prior = self.prior_mlp(deter_state.clone())
        _, prior_stats = self.zdist(prior)

        # Post
        x = torch.cat([deter_state, embedding], dim=-1)
        post = self.post_mlp(x)
        post_dist, post_stats = self.zdist(post)
        if sample:
            post_latent = post_dist.rsample().reshape(action.size(0), -1)
        else:
            post_latent = post_dist.mean.reshape(action.size(0), -1)

        return post_stats, prior_stats, {"deter": deter_state, "stoc": post_latent}
    
class RSSM(nn.Module):
    '''RSSM inner module, with many possible prior outputs.'''
    def __init__(self, embed_dim, cfg):
        super().__init__()
        self.cell = RSSMCell(embed_dim, cfg)
        self.feature_dim = self.cell.feature_dim

    @torch.no_grad()
    def observe(self, embedding, action, hidden_state=None):
        """
        Do single step rollout with embedding in model
        """
        B = action.size(0)
        state = self.cell.init_state(B) if not hidden_state else hidden_state
        post, _, state = self.cell.forward(embedding, action, state)
        feature = self.cell.get_feature(state)
        return post, state, feature

    @torch.no_grad()
    def imagine(self, action, hidden_state=None, model_idx=None):
        """
        Do single step rollout without embedding in model
        """
        B = action.size(0)
        state = self.cell.init_state(B) if not hidden_state else hidden_state
        prior, state = self.cell.prior_forward(action, state, model_idx=model_idx)
        feature = self.cell.get_feature(state)
        return prior, state, feature

    def get_ensemble_dist(self, ensemble_stats):
        return [self.cell.get_dist(stats) for stats in ensemble_stats]

    def get_dist(self, stats, detach=False):
        flat_stats = {}
        for k, v in stats.items():
            flattened_seq = flatten_seq(v)  # (T, B, ...) -> (T*B, ...)
            if detach:
                flattened_seq = flattened_seq.detach()
            flat_stats[k] = flattened_seq
        return self.cell.get_dist(flat_stats)

    @torch.no_grad()
    def ensemble_prior_forward(self, action, hidden_state, sample=True):
        """Used to get latent prediction for entire ensemble"""
        priors, states = [], []
        for k in range(self.cell.n_models):
            prior, state = self.cell.prior_forward(
                action, hidden_state, model_idx=k, sample=sample
            )
            priors.append(prior)
            states.append(state)
        return priors, states

    def forward(
        self,
        embeddings,
        actions,
        resets,
        from_prior=False,
        hidden_state=None,
        sample=True,
        return_state=False,
    ):
        B = embeddings.size(1)  # batch size
        reset_masks = ~resets  # (T, B, 1)

        # Initialize state
        state = self.cell.init_state(B) if not hidden_state else hidden_state

        # if return_state:
        #     states = []

        posts, priors, features = [], [], []

        for embed, act, mask in zip(embeddings, actions, reset_masks):

            if from_prior:
                prior, state = self.cell.prior_forward(act, mask, state, sample=sample)
                post = None
            else:
                post, prior, state = self.cell.forward(
                    embed, act, mask, state, sample=sample
                )

            # if return_states:
            #     states.append(state)
            feature = self.cell.get_feature(state)
            posts.append(post)
            priors.append(prior)
            features.append(feature)

        posts = stack_dict(posts)  # {k: (T, B, D)}
        priors = stack_dict(priors)  # {k: (T, B, D)}
        features = torch.stack(features)  # (T, B, D)

        if return_state:
            # return posts, priors, features, stack_dict(states)
            out_state = {}
            for k, v in state.items():
                out_state[k] = v.detach()
            return posts, priors, features, out_state

        return posts, priors, features
    
class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder = Encoder(cfg.encoder)
        self.rssm = RSSM(self.encoder.feature_dim, cfg.rssm)
        self.decoder = Decoder(self.rssm.feature_dim, self.encoder.repr_dim, cfg.decoder)
        
        # add reward head and other things for kl balancing etc.