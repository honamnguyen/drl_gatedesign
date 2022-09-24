import numpy as np
import argparse, os
from datetime import date
from tqdm import tqdm


from drl.infrastructure.utils import *
from drl.infrastructure.logger import rllib_log_creator

import ray
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.tune.registry import register_env


def transmon_env_creator(kw):
    import gym
    import gym_transmon_cont
    return gym.make('transmon-cont-v7',**kw)

# ray.init(
#     dashboard_host="0.0.0.0",
#     dashboard_port=40001,
# )

if __name__ == "__main__":
    
#     register_env('transmon-cont-v7', lambda kw: gym.make('transmon-cont-v7',**kw))    
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
    config = config.evaluation(
        evaluation_interval = args.testiters,
        evaluation_duration = args.testcount,
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
    
    # For logging
    save_path = '../../../data/'
    os.makedirs(save_path, exist_ok=True)
    run = f'{args.targetgate}_{args.study}_{str(np.random.randint(10000)).zfill(4)}_'
    print(f'\n---> Run {run}\n')

    write_dict_to_file(f'{save_path}{date.today()}_{run}config',vars(args))
    trainer = config.build(
        logger_creator = rllib_log_creator(os.path.expanduser(save_path+'ray_results'), run)
    )
    
    for i in tqdm(range(args.numiter)):
            result = trainer.train()
            # print(result)
            if i*args.stepsperiter % int(1e6) == 0:
                trainer.save()
    # ray.shutdown()


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