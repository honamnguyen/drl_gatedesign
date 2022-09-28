#!bin/bash -x
eval "$(conda shell.bash hook)"
conda activate julius64

python rl_train.py -study checkpointpath -seed 167 -numworkers 8 -numiter 10000