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
    parser.add_argument('-chpt',default='0000',help='Checkpoint. Default: 0000.')
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
    config['env_config']['step_params']['reward_scheme'] = 'local-fidelity-difference-nli'
    config['env_config']['step_params']['reward_type'] =  'worst'
    env = transmon_env_creator(config['env_config'])

    ind = np.array(config['env_config']['channels'][::2])//2
    channels = np.array(['d0','u01','d1','u10'])[ind]
    for checkpoint in glob.glob(f'{run}/checkpoint*'):
        if args.chpt not in checkpoint:
            continue
        agent.restore(checkpoint)        
        done = False
        obs = env.reset()
        data = {
            'channels': channels,
            'pulse': [],
            'avg_fids': [env.avg_fid],
            'worst_fids': [env.fid],
        }
        while not done:
            action = agent.compute_single_action(obs)
            obs, reward, done, _ = env.step(action)
            data['pulse'].append(env.prev_action)
            data['avg_fids'].append(env.avg_fid)
            data['worst_fids'].append(env.fid)     
        data['pulse'] = np.array(data['pulse'])
        data['avg_fids'] = np.array(data['avg_fids'])
        data['worst_fids'] = np.array(data['worst_fids'])
        pickle.dump(data, open(checkpoint.replace('checkpoint',f'RLPulse_{env.fid:.4f}')+'.pkl', 'wb') )
        # np.save(checkpoint.replace('checkpoint',f'pulse_nli{episode_reward:.3f}')+'.npy', np.array(actions))
