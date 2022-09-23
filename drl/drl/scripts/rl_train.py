import numpy as np
import argparse, os
from datetime import date
from tqdm import tqdm

import gym
import gym_transmon_cont
from drl.infrastructure.utils import *
from drl.infrastructure.logger import rllib_log_creator

import ray
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.tune.registry import register_env
# from ray.rllib.models import MODEL_DEFAULTS


# model_config = MODEL_DEFAULTS.copy()
# model_config["fcnet_hiddens"] = [128, 128]
# model_config["fcnet_activation"] = "tanh"

# trainer = DQNTrainer(env=SingleQBit,
#                      config={
#                             "num_workers": 0,
#                             "num_envs_per_worker": 16,
#                             "num_gpus": 0,
#                             "num_cpus_per_worker": 1,
#                             "noisy": True,
#                             "sigma0": 0.5,
#                             "dueling": True,
#                             "hiddens": [128, 128,]
#                             "gamma": 0.99,
#                             "train_batch_size": 200,
#                             "rollout_fragment_length": 200,
#                             "framework": "torch",
#                             "horizon": 100,
#                             "seed": 42,
#                      })

# trainer = PPOTrainer(env=SingleQBit,
#                      config={
#                          "num_workers": 0,
#                          "num_cpus_for_driver": 8,
#                          "num_envs_per_worker": 4,
#                          "num_gpus": 0,
#                          "num_cpus_per_worker": 1,
#                          "sgd_minibatch_size": 1000,
#                          "lr": 0.0005,
#                          'batch_mode': 'truncate_episodes',
#                          "num_sgd_iter": 10,
#                          "horizon": 1000,
#                          "framework": "torch",
#                          "seed": 42,
#                          "model": model_config,
#                      })

# weights = torch.load("pretrained.pth")
# trainer.set_weights({"default_policy": weights})

# print(trainer.get_weights())

ray.init(
    dashboard_host="0.0.0.0",
    dashboard_port=40001,
)

if __name__ == "__main__":
    
    register_env('transmon-cont-v7', lambda kw: gym.make('transmon-cont-v7',**kw))    
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
    config = config.evaluation(
        evaluation_interval = args.testiters,
        evaluation_duration = args.testcount,
    )
    config = config.environment(
        env = 'transmon-cont-v7',
        env_config = transmon_kw(args),
    )
    config.min_sample_timesteps_per_iteration = 1
    
    # For logging
    save_path = '/Users/honamnguyen/MEGA/Berkeley/RL Pulse Optimization/Code/DQN_pulse/drl_gatedesign/data/'
    os.makedirs(save_path, exist_ok=True)
    run = f'{args.targetgate}_{args.study}_{str(np.random.randint(10000)).zfill(4)}_'
    print(f'\n---> Run {run}\n')

    write_dict_to_file(f'{save_path}{date.today()}_{run}config',vars(args))
    trainer = config.build(
        logger_creator = rllib_log_creator(os.path.expanduser(save_path+'ray_results'), run)
    )
    # trainer = config.build()
    
    for i in tqdm(range(int(1e3))):
        result = trainer.train()
        # print(result)
        if i % 50 == 0:
            trainer.save()
    ray.shutdown()


# # ``Tuner.fit()`` allows setting a custom log directory (other than ``~/ray-results``)
# results = ray.tune.Tuner(
#     ppo.PPO,
#     param_space=config,
#     run_config=air.RunConfig(
#         local_dir=log_dir,
#         stop=stop_criteria,
#         checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
#     )).fit()

# # list of lists: one list per checkpoint; each checkpoint list contains
# # 1st the path, 2nd the metric value
# checkpoints = analysis.get_trial_checkpoints_paths(
#     trial=analysis.get_best_trial("episode_reward_mean"),
#     metric="episode_reward_mean")

# # or simply get the last checkpoint (with highest "training_step")
# last_checkpoint = analysis.get_last_checkpoint()
# # if there are multiple trials, select a specific trial or automatically
# # choose the best one according to a given metric
# last_checkpoint = analysis.get_last_checkpoint(
#     metric="episode_reward_mean", mode="max"
# )