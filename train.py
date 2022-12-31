import torch
import hydra
import metaworld

from pathlib import Path
from hydra.utils import to_absolute_path
from utils import *

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
        mt1 = metaworld.MT10(self.cfg.env)
        self.train_env = mt1.train_classes[self.cfg.env]()
        self.train_env.set_task(self.cfg.subtask)
        
        self.model_dir = Path(to_absolute_path('pretrained_models')) / self.cfg.env / self.cfg.subtask / self.cfg.model_type
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.policy_dir = Path(to_absolute_path('pretrained_policies')) / self.cfg.env / self.cfg.subtask / self.cfg.learner
        self.policy_dir.mkdir(parents=True, exist_ok=True)