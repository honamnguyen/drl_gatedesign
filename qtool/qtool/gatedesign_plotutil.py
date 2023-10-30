import glob, pickle

def get_data(run, chpt, data_path, rl=True, verbose=True):
    if rl:
        file = glob.glob(f'{data_path}/ray_results/*{run}*/RLPulse*map*{chpt}*')
        if len(file) == 0:
            file = glob.glob(f'{data_path}/old_ray_results/*{run}*/RLPulse*map*{chpt}*')
    else:
        file = glob.glob(f'{data_path}/two_qubit_gate/theory_{run}*{chpt}.pkl')

    if len(file) != 1:
        print(f'{data_path}/ray_results/*{run}*/RLPulse*map*{chpt}*')
        print(file)
        raise ValueError
    if verbose: print(file[0])
    if rl:
        numpoints = 1
        for key in file[0].split('_'):
            if 'dur' in key:
                duration = int(key.replace('dt','').replace('dur',''))
            elif 'seg' in key:
                numseg = int(key.replace('seg',''))
            elif 'highres' in key:
                numpoints = int(key.replace('.pkl','').replace('highres',''))
        dt = 2/9*duration/numseg/numpoints
    else:
        dt = 2/9
        
    return pickle.load(open(file[0], 'rb')), dt