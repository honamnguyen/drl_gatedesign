import argparse, os, glob, pickle, sys, time
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
    parser.add_argument('-ctrlnoise',type=float,default=1e-2,help='Noisy control variance in %. Default: 1e-2')
    parser.add_argument('-ctrlnoiseparam',default='detune_anharm',help='Noisy control parameters. Default: detune_anharm')
    parser.add_argument('-concat',action=argparse.BooleanOptionalAction,help='Add concat to rl_state for runs before dict obs space. Default: None')
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
    config['env_config']['qsim_params']['ctrl_noise_param'] = 'all'
    if args.concat: 
        config['env_config']['rl_state'] += '_concat'
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
    # param = 'anharm' #'coupling'
    # params = ['detune0'] #'drive_detune_anharm_coupling'
    params = args.ctrlnoiseparam.split('_')
    start = time.time()
    env = transmon_env_creator(config['env_config'])
    for param in params: 
        data[param] = {
            'values': [],
            'avg_fids': [],
            'worst_fids': [],
            'pulses': []
        }
        if param[-1].isdigit():
            data[param]['fiducial'] = config['env_config']['qsim_params']['ctrl'][param[:-1]][int(param[-1])]
        else:
            data[param]['fiducial'] = config['env_config']['qsim_params']['ctrl'][param]


        print(f'\n Fiducial values for {param}:',data[param]['fiducial']/2/np.pi/1e6)
        # Loop over variations in physical parameters
        param_range = np.linspace(1-args.ctrlnoise,1+args.ctrlnoise,51)
        for factor in param_range:
            env_config = deepcopy(config['env_config'])
            val = data[param]['fiducial']*factor
            print('   ',val/2/np.pi/1e6)
            if param[-1].isdigit():
                env_config['qsim_params']['ctrl'][param[:-1]][int(param[-1])] = val
            else:
                env_config['qsim_params']['ctrl'][param] = val
            data[param]['values'].append(val)
            env.sim.reset_ctrl(env_config['qsim_params'])
            # env = transmon_env_creator(env_config)

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
        data[param]['range'] = param_range
        print('\n Fiducial value:',data[param]['fiducial'])
        print(data[param]['avg_fids'])
        pickle.dump(data, open(checkpoint.replace('checkpoint',f'RL_variation{args.ctrlnoise}_{"_".join(params)}')+'.pkl', 'wb') )
    print(f'\n Took {time.time()-start:.3f} sec')
