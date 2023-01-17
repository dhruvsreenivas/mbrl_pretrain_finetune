from gym import envs

envs.register(
    id='AntAngle-v0',
    entry_point='envs.d4rl.ant_angle:AntAngleEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0
)

envs.register(
    id='HalfCheetahJump-v0',
    entry_point='envs.d4rl.cheetah_jump:HalfCheetahJumpEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0
)