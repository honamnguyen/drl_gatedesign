#!bin/bash -x
eval "$(conda shell.bash hook)"
conda activate julius

python rl_train.py -numworkers 4 -study valencia -targetgate ZXp90 -drivestrength 20.47,204.74,15.85,158.54 -anharmonicity ' -310.54,-313.86' -detuning ' -86.66,0' -coupling 2.21