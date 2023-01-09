import warnings
warnings.filterwarnings("ignore")

import torch
import hydra
import metaworld
from tqdm import trange
import wandb

from pathlib import Path
from hydra.utils import to_absolute_path
from utils import *

from mbrl.model import *
from mbrl.rl import *

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = Path.cwd()
        print(f'Workspace directory: {self.work_dir}')
        
        self.setup()
        print('Setup complete.')
        
        self.global_step = 0
        
    def setup(self):
        self.device = torch.device(self.cfg.device)
        set_seed_everywhere(self.cfg.seed)
        
        # create env + set observation/action shapes (at this point choose a random task each time)
        mt10 = metaworld.MT50()
        self.train_env = mt10.train_classes[self.cfg.task]()
        compatible_subtasks = [subtask for subtask in mt10.train_tasks if subtask.env_name == self.cfg.task]
        assert self.cfg.subtask in compatible_subtasks, "Incompatible subtask."
        self.train_env.set_task(self.cfg.subtask)
        self.cfg.state_dim = self.train_env.observation_space.shape
        self.cfg.action_dim = self.train_env.action_space.shape[0]
        
        self.model_dir = Path(to_absolute_path('pretrained_models')) / self.cfg.task / self.cfg.subtask / self.cfg.model_type
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent_dir = Path(to_absolute_path('pretrained_policies')) / self.cfg.task / self.cfg.subtask / self.cfg.agent
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        
        # dataset init
        
        # model + agent init
        if self.cfg.model == 'single_step':
            self.model = SingleStepModel(self.cfg).to(self.device)
        elif self.cfg.model == 'world_model':
            self.model = WorldModel(self.cfg).to(self.device)
        else:
            raise NotImplementedError("Haven't implemented said model yet.")
        
        if self.cfg.agent == 'td3':
            self.agent = TD3(self.cfg).to(self.device)
        elif self.cfg.agent == 'sac':
            self.agent = SAC(self.cfg).to(self.device)
            
        # subconfigs
        self.model_train_cfg = self.cfg.model_training
        
    def train_model(self):
        for _ in trange(1, self.model_train_cfg.epochs + 1):
            # TODO sample batch from dataset, train on batch, etc.
            pass