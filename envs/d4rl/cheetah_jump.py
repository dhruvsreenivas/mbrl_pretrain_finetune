import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahJumpEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        
        v_x = (xposafter - xposbefore) / self.dt
        v_x = np.max(v_x, 3.0) # this is MOPO reward from appendix: https://arxiv.org/pdf/2005.13239.pdf pg 18 to train behavioral policy
        reward_ctrl = -0.1 * np.square(action).sum()
        reward = reward_ctrl + v_x
        done = False
        return ob, reward, done, dict(reward_run=v_x, reward_ctrl=reward_ctrl)
    
    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )
    
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5