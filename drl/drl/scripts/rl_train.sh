#!bin/bash -x
eval "$(conda shell.bash hook)"
conda activate julius64

# python rl_train.py -numworkers 4 -study valencia_test -targetgate ZXp90 -drivestrength 20.47,204.74,15.85,158.54 -anharmonicity ' -310.54,-313.86' -detuning ' -86.66,0' -coupling 2.21

# python rl_train.py -numworkers 0 -study IBMvalencia_q10_sub0.15_workers0_seed167 -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.15 -seed 167

python rl_train.py -numworkers 0 -study SeedTest_IBMvalencia_q1_sub0.15_workers0_seed167 -targetgate X90 -numtransmon 1 -numseg 10 -fidthreshold 0.9999 -worstfidmethod SCQP-dm-0 -channels 1 -duration 10 -IBMbackend valencia -IBMqubits 1 -subactionscale 0.4 -seed 167 -replayinitial 5000