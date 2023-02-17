import argparse, os, glob, pickle, sys
from copy import deepcopy
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
    # parser.add_argument('-env',action=argparse.BooleanOptionalAction,help='Store unitary map or not. Default: None')
    args = parser.parse_args()

    ### ----- LOAD CONFIG + UPDATE----- ###
    ray_path = '../../../data/ray_results/'
    config_file = glob.glob(f'{ray_path}*{args.run}*/params.pkl')[0]
    config = pickle.load(open(config_file, "rb"))
    config['num_workers'] = 0
    config['logger_config'] = {'type': 'ray.tune.logger.NoopLogger'}
    
    # update env config appropriately
    config['env_config']['step_params']['reward_scheme'] = 'local-fidelity-difference-nli'
    config['env_config']['step_params']['reward_type'] =  'worst'
    config['env_config']['qsim_params']['ctrl_noise'] = 0
    config['env_config']['qsim_params']['ctrl_update_freq'] = 'everyepisode'
    ind = np.array(config['env_config']['channels'])
    channels = np.array(['d0','u01','d1','u10'])[ind]
    run = config_file.replace('/params.pkl','')
    
    # Recover checkpoint
    checkpoints = glob.glob(f'{run}/checkpoint*{args.chpt}')
    print(checkpoints)
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    agent = DDPG(config=config)
    agent.restore(checkpoint)   

    data = {}
    param = 'anharm' #'coupling'
    for param in ['drive','detune','anharm','coupling']: 
        data[param] = {
            'fiducial': config['env_config']['qsim_params']['ctrl'][param],
            'values': [],
            'avg_fids': [],
            'worst_fids': [],
            'pulses': []
        }

        print('\n Fiducial value:',config['env_config']['qsim_params']['ctrl'][param])
        # Loop over variations in physical parameters
        for factor in np.linspace(0.99,1.01,11):
            env_config = deepcopy(config['env_config'])
            env_config['qsim_params']['ctrl'][param] *= factor
            data[param]['values'].append(env_config['qsim_params']['ctrl'][param])
            env = transmon_env_creator(env_config)

            obs = env.reset()
            done = False
            pulse = []
            while not done:
                action = agent.compute_single_action(obs)
                obs, reward, done, _ = env.step(action)
                pulse.append(env.prev_action.view(np.complex128))
            data[param]['avg_fids'].append(env.avg_fid)
            data[param]['worst_fids'].append(env.fid)
            data[param]['pulses'].append(pulse)
        data[param]['values'] = np.array(data[param]['values'])
        data[param]['pulses'] = np.array(data[param]['pulses'])
        data[param]['avg_fids'] = np.array(data[param]['avg_fids'])
        data[param]['worst_fids'] = np.array(data[param]['worst_fids'])
        print('\n Fiducial value:',config['env_config']['qsim_params']['ctrl'][param])
        print(data[param]['avg_fids'])
        pickle.dump(data, open(checkpoint.replace('checkpoint',f'RL_variation1e-2_each')+'.pkl', 'wb') )
