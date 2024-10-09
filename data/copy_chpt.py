import argparse, os, glob

from drl.infrastructure.utils import *

if __name__ == "__main__":
    
    ### ----- PARSING ARGUMENTS ----- ###
    args = parser_init(argparse.ArgumentParser()).parse_args()
    ray_path = 'ray_results/'
    
    # log directory
    logdir = glob.glob(f'{ray_path}*{args.targetgate}_{args.study}_*/')
    assert len(logdir) == 1
    logdir = logdir[0]
    print(f'...copy chpts from {logdir}')
    
    # from checkpoint directories
    from_chpt_dirs = glob.glob(f'{logdir}/from*')
    from_chpt_iters = np.array([int(d.split('_')[-1].replace('chpt','')) for d in from_chpt_dirs])
    
    # checkpoint directories to be copied to log directory
    chpt_dirs = glob.glob(from_chpt_dirs[from_chpt_iters.argmax()]+'/checkpoint*')
    
    for chpt_dir in chpt_dirs:
        os.system(f'cp -r {chpt_dir} {logdir}')                           