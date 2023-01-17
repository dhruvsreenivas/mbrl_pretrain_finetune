import warnings
warnings.filterwarnings("ignore")

import torch
import hydra
import metaworld
from tqdm import trange
import wandb
from collections import defaultdict

from pathlib import Path
from hydra.utils import to_absolute_path
from utils import *

from mbrl.model import *
from mbrl.rl import *
from memory import *

D4RL_TASKS = {
    'ant': ['walk', 'angle'],
    'cheetah': ['run', 'jump']
}

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        # subconfigs
        self.model_train_cfg = self.cfg.model_training
        
        self.work_dir = Path.cwd()
        print(f'Workspace directory: {self.work_dir}')
        
        self.setup()
        print('Setup complete.')
        
        self.global_step = 0
        
    def setup(self):
        self.device = torch.device(self.cfg.device)
        set_seed_everywhere(self.cfg.seed)
        
        # create env + set observation/action shapes (at this point choose a random task each time)
        mt50 = metaworld.MT50()
        self.train_env = mt50.train_classes[self.cfg.task]()
        compatible_subtasks = [subtask for subtask in mt50.train_tasks if subtask.env_name == self.cfg.task]
        assert self.cfg.subtask in compatible_subtasks, "Incompatible subtask."
        self.train_env.set_task(self.cfg.subtask)
        self.cfg.state_dim = self.train_env.observation_space.shape
        self.cfg.action_dim = self.train_env.action_space.shape[0]
        
        self.model_dir = Path(to_absolute_path('pretrained_models')) / self.cfg.task / self.cfg.subtask / self.cfg.model_type
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent_dir = Path(to_absolute_path('pretrained_policies')) / self.cfg.task / self.cfg.subtask / self.cfg.agent
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        
        # dataset init
        if self.cfg.task in D4RL_TASKS:
            assert self.cfg.subtask in D4RL_TASKS[self.cfg.task]
            if self.cfg.subtask == D4RL_TASKS[self.cfg.task][1]:
                # meaning we have to grab the other datasets (with other task rewards)
                data_path = f'./offline_data/d4rl/{self.cfg.task}/{self.cfg.level}.npz'
                data = np.load(data_path)
                dataset = OfflineDataset(
                    self.device,
                    data
                )
                self.model_dataloader = dataloader(dataset, self.model_train_cfg.batch_size, self.model_train_cfg.num_workers)
            else:
                data = get_d4rl_dataset(self.cfg.task, self.cfg.level)
                dataset = OfflineDataset(
                    self.device,
                    data
                )
                self.model_dataloader = dataloader(dataset, self.model_train_cfg.batch_size, self.model_train_cfg.num_workers)
        
        # model + agent init
        if self.cfg.model == 'single_step':
            self.model = SingleStepModel(self.cfg).to(self.device)
        elif self.cfg.model == 'world_model':
            self.model = WorldModel(self.cfg).to(self.device)
        else:
            raise NotImplementedError("Haven't implemented said model yet.")
        
        model_kwargs = {'lr': self.model_train_cfg.lr, 'eps': self.model_train_cfg.eps}
        self.model_opt = get_optimizer(self.model_train_cfg.optim, self.model.parameters(), **model_kwargs)
        
        if self.cfg.agent == 'td3':
            self.agent = TD3(self.cfg).to(self.device)
        elif self.cfg.agent == 'sac':
            self.agent = SAC(self.cfg).to(self.device)
        
    def train_model(self):
        '''Pretrains world model offline. Problably should consider pretraining without rewards and finetuning on rewards'''
        for _ in trange(1, self.model_train_cfg.epochs + 1):
            epoch_metrics = defaultdict(AverageMeter)
            for batch in self.model_dataloader:
                loss, metrics = self.model.loss(batch)
                
                # optimizer, store loss
                self.model_opt.zero_grad()
                loss.backward()
                self.model_opt.step()
                
                for k, v in metrics.items():
                    epoch_metrics[k].update(v, 1)
                    
            avgs = {k: v.value() for k, v in epoch_metrics.items()}
            if self.cfg.wandb:
                wandb.log(avgs)
        
        