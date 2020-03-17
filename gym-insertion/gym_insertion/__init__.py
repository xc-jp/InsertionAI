from gym.envs.registration import register

register(
    id='insertion-v0',
    entry_point='gym_insertion.envs:InsertionEnv',
)
