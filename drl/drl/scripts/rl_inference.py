import argparse, os, glob, pickle, sys
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
    parser.add_argument('-map',action=argparse.BooleanOptionalAction,help='Store unitary map or not. Default: None')
    parser.add_argument('-concat',action=argparse.BooleanOptionalAction,help='Add concat to rl_state for runs before dict obs space. Default: None')
    args = parser.parse_args()

    ### ----- LOAD CONFIG + UPDATE----- ###
    ray_path = '../../../data/ray_results/'
    config_file = glob.glob(f'{ray_path}*{args.run}*/params.pkl')[0]
    config = pickle.load(open(config_file, "rb"))
    config['num_workers'] = 0
    config['logger_config'] = {'type': 'ray.tune.logger.NoopLogger'}
    config['env_config']['step_params']['reward_scheme'] = 'local-fidelity-difference-nli'
    config['env_config']['step_params']['reward_type'] =  'worst'
    config['env_config']['qsim_params']['ctrl_noise'] = 0
    config['env_config']['qsim_params']['ctrl_noise_param'] = 'all'
    config['env_config']['qsim_params']['ctrl_update_freq'] = 'everyepisode'
    if args.concat: 
        config['env_config']['rl_state'] += '_concat'
        print(f"\n  rl_state = {config['env_config']['rl_state']} \n")
    run = config_file.replace('/params.pkl','')

    agent = DDPG(config=config)
    env = transmon_env_creator(config['env_config'])
    # print(config['env_config']['qsim_params'])
    # sys.exit()
    
    # ind = np.array(config['env_config']['channels'][::2])//2
    ind = np.array(config['env_config']['channels'])
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
            'leakages': [env.leakage],
        }
        if args.map: 
            data['map'] = [env.map]
            suffix = '_map'
        else:
            suffix = ''
        while not done:
            action = agent.compute_single_action(obs)
            obs, reward, done, _ = env.step(action)
            data['pulse'].append(env.prev_action.view(np.complex128))
            data['avg_fids'].append(env.avg_fid)
            data['worst_fids'].append(env.fid)     
            data['leakages'].append(env.leakage)   
            if args.map: data['map'].append(env.map)     
        data['pulse'] = np.array(data['pulse'])
        data['avg_fids'] = np.array(data['avg_fids'])
        data['worst_fids'] = np.array(data['worst_fids'])
        data['leakages'] = np.array(data['leakages'])
        if args.map: data['map'] = np.array(data['map'])
        name = checkpoint.replace('checkpoint_',f'RLPulse_run{args.run}_{env.fid:.5f}_{env.avg_fid:.5f}{suffix}_ep')+'.pkl'
        pickle.dump(data, open(name, 'wb') )
        # np.save(checkpoint.replace('checkpoint',f'pulse_nli{episode_reward:.3f}')+'.npy', np.array(actions))
