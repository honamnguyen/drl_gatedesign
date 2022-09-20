from gym.envs.registration import register

register(
    id='transmon-cont-v7',
    entry_point='gym_transmon_cont.envs:ContinuousTransmonEnv',
)