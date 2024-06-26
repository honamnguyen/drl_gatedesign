import numpy as np
import argparse, os, glob, pickle
from datetime import date
from tqdm import tqdm

from drl.infrastructure.utils import *
from drl.infrastructure.logger import rllib_log_creator, rllib_log_creator_checkpoint

import ray
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.rllib.algorithms.ddpg import DDPG
from ray.tune.registry import register_env
from ray.rllib.utils.debug.deterministic import update_global_seed_if_necessary

class Seeded_DDPG(DDPG):
    def __init__(self, config, logger_creator=None):
        if logger_creator:
            super().__init__(config=config,logger_creator=logger_creator)
        else:
            super().__init__(config=config)
        if config['seed'] is not None:
            print(f"\n-  Set global seed to {config['seed']} in DDPG  -\n")
            update_global_seed_if_necessary(config['framework'],config['seed'])

def transmon_env_creator(kw):
    import gym
    import gym_transmon_cont
    return gym.make('transmon-cont-v7',**kw)

if __name__ == "__main__":
    
#     register_env('transmon-cont-v7', lambda kw: gym.make('transmon-cont-v7',**kw))    
    register_env('transmon-cont-v7', transmon_env_creator)    
    ### ----- PARSING ARGUMENTS ----- ###
    args = parser_init(argparse.ArgumentParser()).parse_args()
    ray_path = '../../../data/ray_results/'
    
    if args.chpt != '':
        
        config_file = glob.glob(f'{ray_path}*{args.targetgate}*{args.study}*/params.pkl')
        checkpoint = glob.glob(f'{ray_path}*{args.targetgate}*{args.study}*/checkpoint_{args.chpt.zfill(6)}')
        print('  --config_file: ',config_file)
        print('  --checkpoint: ',checkpoint)
        assert len(config_file) == 1 and len(checkpoint) == 1
        istart = int(args.chpt)
        # chpt_iteration = np.array([int(c.split('_')[-1]) for c in checkpoint])
        # istart = chpt_iteration.max()
        
        # get run name
        logdir = config_file[0].replace('/params.pkl','')
        logger_creator = rllib_log_creator_checkpoint(logdir+f'/from_chpt{str(istart).zfill(6)}'+args.rstudy) 
        config = pickle.load(open(config_file[0], "rb"))
        config['exploration_config']['random_timesteps'] = args.randomtimesteps
        # config['exploration_config']['initial_scale'] = 0.1
        
        ## need time to make this restart change more generally ##
        if args.rctrlnoise is not None:
            config['env_config']['qsim_params']['ctrl_noise'] = args.rctrlnoise
        if args.rctrlnoisedist is not None:
            config['env_config']['qsim_params']['ctrl_noise_dist'] = args.rctrlnoisedist
        if args.rseed is not None:
            config['seed'] = int(args.rseed)
        if args.drift is not None:        
            variation = pickle.load(open(f'../../../data/{args.drift}.pkl', 'rb'))
            config['env_config']['qsim_params']['fixed_variation'] = variation
        
        
        # env_config = transmon_kw(args)
        # for key in env_config.keys():
        #     print(env_config[key])
        #     cond = (not np.array_equal(env_config[key],config['env_config'][key])) if type(env_config[key]) is np.ndarray else (env_config[key] != config['env_config'][key])
        #     if cond:
        #         print('-* Replace ',config['env_config'][key])
        #         print('-* with ',env_config[key])
        #         config['env_config'][key] = env_config[key]
        
        trainer = Seeded_DDPG(config=config, logger_creator=logger_creator)
        trainer.restore(checkpoint[0])
        print(f'\n---> Restart run from {checkpoint[0]}\n')
        # trainer.restore(checkpoint[chpt_iteration.argmax()])
        # print(f'\n---> Restart run from {checkpoint[chpt_iteration.argmax()]}\n')
        
        
    else:
        istart = 0
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
        # if args.td3policydelay != -1:
        #     config = config.training(
        #         twin_q = True,
        #         policy_delay = args.td3policydelay,
        #         smooth_target_policy = True,
        #     ) 
        # if args.td3policydelay != -1:
        config = config.training(
            twin_q = args.td3twinq,
            policy_delay = args.td3policydelay,
            smooth_target_policy = args.td3smoothtarget,
        )   
        config = config.evaluation(
            evaluation_interval = args.evaluationinterval,
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
            num_rollout_workers = args.numworkers,
            recreate_failed_workers = True,
            restart_failed_sub_environments=True,
        )
        config = config.resources(
            num_gpus = args.numgpus
        )

        if args.seed:
            config = config.debugging(
                seed=args.seed
            )

        # For logging
        os.makedirs(ray_path, exist_ok=True)
        run = f'{args.targetgate}_{args.study}_{str(np.random.randint(10000)).zfill(4)}_'
        print(f'\n---> Run {run}')
        if args.IBMbackend:
            print(f'-  Use configuration from IBM backend: {args.IBMbackend}')
        print()

        # convert to config dict to pass to ALGO
        config = config.to_dict()
        logger_creator = rllib_log_creator(os.path.expanduser(ray_path), run)
        trainer = Seeded_DDPG(config=config, logger_creator=logger_creator)   
        trainer.save() # save the initial point
        
    for i in tqdm(range(istart, args.numiter)):
            result = trainer.train()
            # print(result)
            if (i+1) % args.evaluationinterval == 0:
                trainer.save()

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