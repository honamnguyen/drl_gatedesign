import argparse, os, glob, pickle, numpy as np
from tqdm import tqdm
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def eventfile_to_dict(file,filters=['reward','episodes_','step']):
    output = {}
    for e in tqdm(EventFileLoader(file).Load()):
        for v in e.summary.value:
            if any([key in v.tag for key in filters]):
                if v.tag in output:
                    output[v.tag].append(v.tensor.float_val)
                else:
                    output[v.tag] = [v.tensor.float_val]
    return output

if __name__ == '__main__':

    ### ----- PARSING ARGUMENTS ----- ###   
    parser = argparse.ArgumentParser()
    parser.add_argument('-runs',help='Runs to save event log.')
    parser.add_argument('-old',default='',help='Look into old_ray_results.')
    args = parser.parse_args()

    for run in args.runs.split(','):
        ray_path = f'../../../data/{args.old}ray_results/'
        param_file = glob.glob(f'{ray_path}*{run}*/params.pkl')[0]
        event_files = glob.glob(f'{ray_path}*{run}*/events*')

        if len(event_files) != 1:
            print(f'{ray_path}*{run}*/events*')
            raise ValueError

        log = eventfile_to_dict(event_files[0])
        data = {}
        for key in log.keys():
            val = np.array(log[key]).flatten()
            if 0 not in val.shape:
                data[key.replace('ray/tune/','')] = val
        pickle.dump(data, open(param_file.replace('params','eventlog'), 'wb'))