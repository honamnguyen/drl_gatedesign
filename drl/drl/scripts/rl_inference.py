import argparse, os, glob, pickle
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from drl.infrastructure.utils import *

from ray.tune.registry import register_env
from ray.rllib.algorithms.ddpg import DDPGConfig, DDPG
        
def transmon_env_creator(kw):
    import gym
    import gym_transmon_cont
    return gym.make('transmon-cont-v7',**kw)

if __name__ == '__main__':

    register_env('transmon-cont-v7', transmon_env_creator)    
    ### ----- PARSING ARGUMENTS ----- ###   
    parser = argparse.ArgumentParser()
    parser.add_argument('-run',default='0000',help='Name fragments of run. Default: 0000.')
    args = parser.parse_args()

    ### ----- LOAD CONFIG + UPDATE----- ###
    ray_path = '../../../data/ray_results/'
    config_file = glob.glob(f'{ray_path}*{args.run}*/params.pkl')[0]
    config = pickle.load(open(config_file, "rb"))
    config['num_workers'] = 0
    config['logger_config'] = {'type': 'ray.tune.logger.NoopLogger'}
    # for key in config:
    #     print(key,config[key])
    agent = DDPG(config=config)
    
    # INFERENCE   
    run = config_file.replace('/params.pkl','')
    env = transmon_env_creator(config['env_config'])

    for checkpoint in glob.glob(f'{run}/checkpoint*'):
        agent.restore(checkpoint)        
        episode_reward = 0
        done = False
        obs = env.reset()
        actions = []
        while not done:
            action = agent.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            actions.append(action)
        print(f'\n{checkpoint.split("/")[-1]}: {episode_reward}')