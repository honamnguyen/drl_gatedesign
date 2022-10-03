import numpy as np
import argparse, os
from datetime import date
import glob

from drl.infrastructure.utils import *
from drl.infrastructure.logger import rllib_log_creator

import ray
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.rllib.algorithms import ddpg
from ray.tune.registry import register_env


def transmon_env_creator(kw):
    import gym
    import gym_transmon_cont
    return gym.make('transmon-cont-v7',**kw)


if __name__ == "__main__":
    
    register_env('transmon-cont-v7', transmon_env_creator)    
    ### ----- PARSING ARGUMENTS ----- ###
    args = parser_init(argparse.ArgumentParser())
    
    config = DDPGConfig().framework('torch')
    
    config = config.training(
        gamma = args.gamma,
        actor_lr = args.actorlr,
        critic_lr = args.criticlr,
        train_batch_size = args.batchsize,
        replay_buffer_config = {
            'capacity': args.replaysize,
            'learning_starts': args.replayinitial,
        },
        actor_hiddens = [int(x) for x in args.hidsizes.split(',')],
        critic_hiddens = [int(x) for x in args.hidsizes.split(',')],
        actor_hidden_activation = args.activation,
        critic_hidden_activation = args.activation,
    )
    config = config.exploration(
        explore = False,
    )
    config = config.evaluation(
        evaluation_interval = args.evaluationinterval,
        evaluation_duration = args.testcount,
        # evaluation_config = {'explore': False}
    )
    config = config.environment(
        env = 'transmon-cont-v7',
        env_config = transmon_kw(args),
    )
    
    config = config.reporting(
        min_sample_timesteps_per_iteration = args.stepsperiter
    )
    config = config.rollouts(
        num_rollout_workers = args.numworkers
    )
    if args.seed:
        config = config.debugging(
            seed=args.seed
        )
    ray_path = '/Users/honamnguyen/MEGA/Berkeley/RL_Pulse_Optimization/Code/DQN_pulse/drl_gatedesign/data/ray_results/'
    
    # ray_path = '$drl_gatedesign/data/ray_results/'
   
    agent = ddpg.DDPG(config=config)
    env = transmon_env_creator(transmon_kw(args))
    print('\n------Untrained-----')
    
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    print(episode_reward)
    
    # checkpoint_path = f'{ray_path}2022-09-23_sqrtZX_NoStudy_9456_52fm55le/checkpoint_004001'
    
    # print(glob.glob(f'{ray_path}*{args.checkpointpath}*/checkpoint*'))
    print('\n------Trained-----')
    for checkpoint in glob.glob(f'{ray_path}*{args.checkpointpath}*/checkpoint*'):
        agent.restore(checkpoint)        
        for i in range(3):
            print(f'--i={i}--')
            episode_reward = 0
            done = False
            obs = env.reset()
            rewards = []
            while not done:
                action = agent.compute_single_action(obs)
                # print(action)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                rewards.append(reward)
            # print(rewards)
            print(f'{checkpoint.split("/")[-1]}: {episode_reward}')
                         