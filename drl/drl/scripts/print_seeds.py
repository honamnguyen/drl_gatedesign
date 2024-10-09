import numpy as np
import glob
import argparse
from datetime import date
def get_seeds_from_config(fname,var1,var2,range1,range2):
    '''
    Return an array of seeds corresponding to the data files for a particular sweep over 2 variables.
    '''
    all_seeds = np.zeros([len(range1),len(range2)],dtype=int)

    for file in glob.glob(fname):
        i,j=None,None
        with open(file) as f:
            for line in f:
                key,val = line.rstrip().split(': ')
                if key==var1:
                    i = range1.index(val)
                if key==var2:
                    j = range2.index(val)
        all_seeds[i,j] = int(file.split('_')[-2])
    return all_seeds

if __name__ == "__main__":

    ### ----- PARSING ARGUMENTS ----- ###
    parser = argparse.ArgumentParser()
    parser.add_argument('-date',default=date.today(),help='Date.')
    parser.add_argument('-gate',default='sqrtZX',help='Target gate.')
    parser.add_argument('-study',help='Study.')
    parser.add_argument('-algo',default='ddpg',help='RL algorithm.')
    parser.add_argument('-var1',help='Varible 1.')
    parser.add_argument('-var2',help='Varible 2.')
    parser.add_argument('-range1',default='20_30_40',help='Range 1')
    parser.add_argument('-range2',default='800,400,200_400,200,100',help='Range 2')        
    args = parser.parse_args()
    
    date = args.date
    gate = args.gate
    study = args.study
    algo = args.algo
    var1 = args.var1
    var2 = args.var2
    range1 = args.range1.split('_')
    range2 = args.range2.split('_')

    if var1=='targetgate' or var2=='targetgate':
        run = f'../../../data/{date}_*_{study}'
    else:
        run = f'../../../data/{date}_{gate}_{study}'
    print(run)
    all_seeds = get_seeds_from_config(run+'_*config.txt',var1,var2,range1,range2)

    print(f'\nrow {var1}: {range1}')
    print(f'col {var2}: {range2}')
    print(all_seeds)