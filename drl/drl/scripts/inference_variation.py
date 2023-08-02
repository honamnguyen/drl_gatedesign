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
    parser.add_argument('-normalizedcontext',action=argparse.BooleanOptionalAction,help='Normalize context variable for run before July 2023.')
    parser.add_argument('-avgfid',action=argparse.BooleanOptionalAction,help='Fast inference with avgfid only. Default: None')
    parser.add_argument('-npoints',type=int,default=51,help='Number of points to evaluate. Default: 51')


    args = parser.parse_args()

    ### ----- LOAD CONFIG + UPDATE----- ###
    ray_path = '../../../data/ray_results/'
    config_file = glob.glob(f'{ray_path}*{args.run}*/params.pkl')[0]
    config = pickle.load(open(config_file, "rb"))
    config['num_workers'] = 0
    config['logger_config'] = {'type': 'ray.tune.logger.NoopLogger'}
    
    # update env config appropriately
    if config['env_config']['rl_state'] == 'ket_detune_0':
        config['env_config']['rl_state'] = 'ket_detune0'
    if config['env_config']['qsim_params']['ctrl_noise'] == 0:
        config['env_config']['qsim_params']['ctrl_noise'] = args.ctrlnoise # for runs trained on fixed env
        
    if args.avgfid:
        suffix = '_avgfid'
    else:
        suffix = ''
        config['env_config']['step_params']['reward_scheme'] = 'local-fidelity-difference-nli'
        config['env_config']['step_params']['reward_type'] =  'worst'
    config['env_config']['qsim_params']['ctrl_noise_param'] = args.ctrlnoiseparam
    config['env_config']['qsim_params']['ctrl_update_freq'] = 'everyepisode'
    
    # Backwards compatibility
    if args.concat: 
        config['env_config']['rl_state'] += '_concat'
    if args.normalizedcontext:
        config['env_config']['normalized_context'] = True

        
    ind = np.array(config['env_config']['channels'])
    channels = np.array(['d0','u01','d1','u10'])[ind]
    run = config_file.replace('/params.pkl','')
    
    env = transmon_env_creator(config['env_config'])
    
    data_temp = {
        'variations': np.linspace(-args.ctrlnoise,args.ctrlnoise,args.npoints),
        'fiducials':{},
        'avg_fids': [],
        'worst_fids': [],
        'pulses': []
    }
    params = args.ctrlnoiseparam.split('_')
    for param in params:
        if param[-1].isdigit():
            data_temp['fiducials'][param] = config['env_config']['qsim_params']['ctrl'][param[:-1]][int(param[-1])]
        else:
            data_temp['fiducials'][param] = config['env_config']['qsim_params']['ctrl'][param]
        print(f"Fiducial values for {param}:",(data_temp['fiducials'][param]/2/np.pi/1e6).round(3)," MHz")
        
    # Recover checkpoint
    if 'range' in args.chpt:
        vmin, vmax, step = [int(x) for x in args.chpt.replace('range_','').split('_')]
        chpts = np.arange(vmin,vmax+step,step)[::-1]
    else:
        chpts = [int(x) for x in args.chpt.split('_')]
    for chpt in chpts:
        chpt = str(int(100*chpt)).zfill(6)
        start = time.time()
        checkpoints = glob.glob(f'{run}/checkpoint*{chpt}')
        print(checkpoints)
        assert len(checkpoints) == 1
        checkpoint = checkpoints[0]
        agent = DDPG(config=config)
        agent.restore(checkpoint)   
        
        data = deepcopy(data_temp)
        for variation in data['variations']:
            done = False
            variation_dict = {}
            for param in params:
                if param[-1].isdigit():
                    variation_dict[param] = np.array([variation]) 
                else:
                    variation_dict[param] = variation*np.ones(len(env.sim.ctrl[param]))
            obs = env.reset(variation=variation_dict)
            # obs = env.reset(variation={param:np.array([variation]) for param in params})
            pulse = []
            while not done:
                action = agent.compute_single_action(obs)
                obs, reward, done, _ = env.step(action)
                pulse.append(env.prev_action.view(np.complex128))
            data['avg_fids'].append(env.avg_fid)
            data['worst_fids'].append(env.fid)
            data['pulses'].append(np.array(pulse))
            print(f'var = {variation:.3f}, {env.avg_fid:.5f}')
        data['pulses'] = np.array(data['pulses'])
        data['avg_fids'] = np.array(data['avg_fids'])
        data['worst_fids'] = np.array(data['worst_fids'])
        print(data['avg_fids'])
        pickle.dump(data, open(checkpoint.replace('checkpoint',f'RLGen_variation{args.ctrlnoise}_{"_".join(params)}{suffix}')+'.pkl', 'wb') )
        print(f'\n Took {time.time()-start:.3f} sec')

        # param_range = np.linspace(1-args.ctrlnoise,1+args.ctrlnoise,51)
        # for factor in param_range:
        #     env_config = deepcopy(config['env_config'])
        #     val = data[param]['fiducial']*factor
        #     print('   ',val/2/np.pi/1e6)
        #     if param[-1].isdigit():
        #         env_config['qsim_params']['ctrl'][param[:-1]][int(param[-1])] = val
        #     else:
        #         env_config['qsim_params']['ctrl'][param] = val
        #     data[param]['values'].append(val)
        #     env.sim.reset_ctrl(env_config['qsim_params'])

            # obs = env.reset()
            # done = False
            # pulse = []
            # while not done:
            #     action = agent.compute_single_action(obs)
        #         obs, reward, done, _ = env.step(action)
        #         pulse.append(env.prev_action.view(np.complex128))
        #     data[param]['avg_fids'].append(env.avg_fid)
        #     data[param]['worst_fids'].append(env.fid)
        #     data[param]['pulses'].append(pulse)
        # data[param]['values'] = np.array(data[param]['values'])
        # data[param]['pulses'] = np.array(data[param]['pulses'])
        # data[param]['avg_fids'] = np.array(data[param]['avg_fids'])
        # data[param]['worst_fids'] = np.array(data[param]['worst_fids'])
        # data[param]['range'] = param_range
        # print('\n Fiducial value:',data[param]['fiducial'])
        # print(data[param]['avg_fids'])
        # pickle.dump(data, open(checkpoint.replace('checkpoint',f'RL_variation{args.ctrlnoise}_{"_".join(params)}')+'.pkl', 'wb') )
    # print(f'\n Took {time.time()-start:.3f} sec')
