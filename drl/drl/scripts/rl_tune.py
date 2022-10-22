import numpy as np
import argparse, os, glob
from datetime import date
from tqdm import tqdm

from drl.infrastructure.utils import *
from drl.infrastructure.logger import rllib_log_creator

import ray
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.rllib.algorithms.ddpg import DDPG
from ray.tune.registry import register_env
from ray.rllib.utils.debug.deterministic import update_global_seed_if_necessary

from ray import tune, air
from ray.tune.schedulers import ASHAScheduler

class Seeded_DDPG(DDPG):
    def __init__(self, config, logger_creator):
        super().__init__(config=config,logger_creator=logger_creator)
        if config['seed'] is not None:
            print(f"\n-  Set global seed to {config['seed']} in DDPG  -\n")
            update_global_seed_if_necessary(config['framework'],config['seed'])

def transmon_env_creator(kw):
    import gym
    import gym_transmon_cont
    return gym.make('transmon-cont-v7',**kw)

# ray.init(
#     dashboard_host="0.0.0.0",
#     dashboard_port=40001,
# )

if __name__ == '__main__':
    
#     register_env('transmon-cont-v7', lambda kw: gym.make('transmon-cont-v7',**kw))    
    register_env('transmon-cont-v7', transmon_env_creator)
    

    
    
    ### ----- PARSING ARGUMENTS ----- ###
    args = parser_init(argparse.ArgumentParser()).parse_args()
    config = DDPGConfig().framework('torch').to_dict()
    config.update({
        'framework': 'torch',
        'gamma': args.gamma,
        'actor_lr' : args.actorlr,
        'critic_lr' : args.criticlr,
        'train_batch_size' : args.batchsize,
        'actor_hiddens' : [int(x) for x in args.hidsizes.split(',')],
        'critic_hiddens' : [int(x) for x in args.hidsizes.split(',')],
        'actor_hidden_activation' : args.activation,
        'critic_hidden_activation' : args.activation, 
        'evaluation_interval' : args.evaluationinterval,
        'evaluation_duration' : args.testcount,
        'env' : 'transmon-cont-v7',
        'env_config' : transmon_kw(args),
        'min_sample_timesteps_per_iteration' : args.stepsperiter,
        'num_workers' : args.numworkers,
        'num_gpus': args.tunegpus,
    })
    
    config['replay_buffer_config'].update({
        'capacity': args.replaysize,
        'learning_starts': args.replayinitial,
    })

    if args.td3policydelay != -1:
        config.update({ 
            'twin_q' : True,
            'policy_delay' : args.td3policydelay,
            'smooth_target_policy' : True,
        })   
    if args.seed: config.update({'seed': args.seed})
           
    # For logging
    save_path = '../../../data/'
    os.makedirs(save_path, exist_ok=True)
    run = f'{args.targetgate}_{args.study}_{str(np.random.randint(10000)).zfill(4)}_'
    print(f'\n---> Run {run}')
    if args.IBMbackend:
        print(f'-  Use configuration from IBM backend: {args.IBMbackend}')
    print()

    # convert to config dict to pass to ALGO
    logger_creator = rllib_log_creator(os.path.expanduser(save_path+'ray_results'), run)
    # trainer = Seeded_DDPG(config=config, logger_creator=logger_creator)   
    
    # if args.chptrun:
    #     checkpoint = glob.glob(f'{save_path}ray_results/*{args.chptrun}*/checkpoint*{args.chpt}')
    #     assert len(checkpoint) == 1
    #     trainer.restore(checkpoint[0])
    
    # save the initial point
    # trainer.save()
    # for i in tqdm(range(args.numiter)):
    #         result = trainer.train()
    #         # print(result)
    #         if (i+1) % args.evaluationinterval == 0:
                # trainer.save()
    # ray.shutdown()
    param_space = {  # Hyperparameter space
        'gamma': tune.uniform(0.95, 0.99),
        'actor_lr': tune.uniform(1e-5, 1e-3),
        'actor_hiddens' : [[800,400,200], [400,200,100], [400,200]],
        'critic_hiddens' : [[800,400,200], [400,200,100], [400,200]],
        'critic_lr': tune.uniform(1e-5, 1e-3),
        'train_batch_size': [32, 64, 128, 256],
    }
    # param_space = config.update(param_space)
        
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='episode_reward_mean',
        mode='max',
        max_t=5000,
        grace_period=100)

    tuner = tune.Tuner(
        # tune.with_resources('DDPG', {'cpu': args.tunecpu, 'gpu': args.tunegpu}),
        'DDPG',
        run_config=air.RunConfig(
            name='tune_asha',
            stop={"episode_reward_mean": 3},
            verbose=1,
            local_dir=save_path+'ray_results',
            checkpoint_config=config,
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler, num_samples=1,
        ),
        param_space=param_space,
    )
    
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)


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