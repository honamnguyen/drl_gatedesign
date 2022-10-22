#!bin/bash -x
eval "$(conda shell.bash hook)"
conda activate julius64

python rl_tune.py -numworkers 6 -study IBMvalencia_q10_seed167_sub0.15 -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.15 -seed 167 -tunegpu 0

