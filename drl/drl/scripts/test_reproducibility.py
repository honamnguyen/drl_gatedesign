import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gym
import numpy as np
import unittest

import ray
from ray.rllib.algorithms.dqn import DQN 
from ray.rllib.algorithms.ddpg import DDPG
# from ray.rllib.algorithms.dqn import DQN as ALGO
# from ray.rllib.algorithms.ddpg import DDPG as ALGO
from ray.rllib.algorithms.td3 import TD3 as ALGO

from ray.rllib.utils.test_utils import framework_iterator
from ray.tune.registry import register_env
import torch
import random

# class custom_ALGO(ALGO):
#     def __init__(self, config, env):
#         super().__init__(config=config, env=env)
#         set_reproducibillity(config['seed'])

def set_reproducibillity(seed=None):
    if seed is not None:
        torch.manual_seed(seed)    
        np.random.seed(seed)
        random.seed(seed)

class TestReproducibility(unittest.TestCase):
    def test_reproducing_trajectory(self):
        class PickLargest(gym.Env):
            def __init__(self, config):
                print('-------Intialize environment-------')
                self.observation_space = gym.spaces.Box(
                    low=float("-inf"), high=float("inf"), shape=(4,)
                )
                self.config = config
                if config['discrete']:
                    self.action_space = gym.spaces.Discrete(4)
                else:
                    self.action_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))
            def reset(self, **kwargs):
                self.obs = np.random.randn(4)
                return self.obs

            def step(self, action):
                if self.config['discrete']:
                    reward = self.obs[action]
                else:
                    reward = self.obs@action
                return self.obs, reward, True, {}

        def env_creator(env_config):
            return PickLargest(env_config)

        discrete = True if ALGO == DQN else False
        seeds = np.random.randint(0,10000,size=2)

        for fw in framework_iterator(frameworks=("torch")):
            trajs = list()
            for trial in range(3):
                ray.init()
                register_env("PickLargest", env_creator)
                config = {
                    "seed": int(seeds[0]) if trial in [0, 1] else int(seeds[1]),
                    "min_time_s_per_iteration": 0,
                    "min_sample_timesteps_per_iteration": 100,
                    "framework": fw,
                    'env_config': {'discrete': discrete},
                }
                # set_reproducibillity(config['seed'])
                # agent = DQN(config=config, env="PickLargest")
                # agent = custom_ALGO(config=config, env="PickLargest")
                agent = ALGO(config=config, env="PickLargest")

                trajectory = list()
                for _ in range(8):
                    r = agent.train()
                    trajectory.append(r["episode_reward_max"])
                    trajectory.append(r["episode_reward_min"])
                trajs.append(trajectory)

                ray.shutdown()

            # trial0 and trial1 use same seed and thus
            # expect identical trajectories.
            all_same = True
            for v0, v1 in zip(trajs[0], trajs[1]):
                if v0 != v1:
                    all_same = False
            self.assertTrue(all_same)

            # trial1 and trial2 use different seeds and thus
            # most rewards tend to be different.
            diff_cnt = 0
            for v1, v2 in zip(trajs[1], trajs[2]):
                if v1 != v2:
                    diff_cnt += 1
            self.assertTrue(diff_cnt > 8)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))