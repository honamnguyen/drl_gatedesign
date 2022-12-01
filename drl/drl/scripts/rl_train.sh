#!bin/bash -x
eval "$(conda shell.bash hook)"

# python rl_train.py -numworkers 4 -study valencia_test -targetgate ZXp90 -drivestrength 20.47,204.74,15.85,158.54 -anharmonicity ' -310.54,-313.86' -detuning ' -86.66,0' -coupling 2.21

# python rl_train.py -numworkers 6 -study IBMvalencia_q10_seed167_sub0.15 -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.15 -seed 167

# python rl_train.py -numworkers 4 -study IBMvalencia_q10_seed167_sub0.075 -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.075 -seed 167

# python rl_train.py -numworkers 6 -study valencia_sub0.1 -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -chpt

# python rl_train.py -numworkers 6 -study IBMvalencia_q10_seed167_workers6_chan3_dur15_seg10_sub0.4_avg -targetgate IX90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.4 -seed 167 -duration 15 -numseg 10 -rewardtype average -channel 3

# python rl_train.py -numworkers 6 -study IBMvalencia_q10_seed167_workers6_chan2_dur128dt_seg8_sub0.4 -targetgate IX90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.4 -seed 167 -duration 128 -numseg 8 -channel 2 -evolvemethod TISE

# python rl_train.py -numworkers 6 -study IBMvalencia_q10_seed167_workers6_chan2_dur128dt_seg8_sub0.4 -targetgate X90I -IBMbackend valencia -IBMqubits 10 -subactionscale 0.4 -seed 167 -duration 128 -numseg 8 -channel 0 -evolvemethod TDSE
 
######## ON LINUX ##########
conda activate julius

# python rl_train.py -numworkers 4 -study IBMvalencia_q10_seed167_workers4_chan012_dur1120dt_seg20_sub0.1 -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channel 012 -evolvemethod TDSE

# python rl_train.py -numworkers 4 -study IBMvalencia_q10_seed167_workers4_chan012_dur1120dt_seg20_sub0.1_avg -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channel 012 -evolvemethod TDSE -rewardtype average

python rl_train.py -numworkers 4 -study workers4_IBMvalencia_q10_seed167_chan12_TISE_dur1120dt_seg20_sub0.1_avg -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channel 12 -evolvemethod TISE -rewardtype average

# python rl_train.py -numworkers 6 -study IBMvalencia_q10_seed167_workers6_chan2_dur128dt_seg8_sub0.4 -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 

# python rl_train.py -numworkers 4 -study IBMvalencia_q10_seed167_workers4_chan023_dur1120dt_seg20_sub0.1 -targetgate NOTC -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channel 023 -evolvemethod TDSE


# python rl_train.py -numworkers 4 -study IBMvalencia_q10_seed167_workers4_chan012_dur1120dt_seg20_sub0.1_batch150_delay1 -targetgate NOTCCNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channel 012 -batchsize 150 -td3policydelay 1 -evolvemethod TDSE 

########### ON MAC ###########
# conda activate julius64

# python rl_train.py -numworkers 8 -study workers8_IBMvalencia_q10_seed2_chan12_TISE_dur1120dt_seg20_sub0.1_batch150 -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 2 -duration 1120 -numseg 20 -channel 12 -batchsize 150 -evolvemethod TISE 

# python rl_train.py -numworkers 4 -study workers4_IBMvalencia_q10_seed2_chan12_TISE_dur1120dt_seg20_sub0.1_batch150 -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 2 -duration 1120 -numseg 20 -channel 12 -batchsize 150 -evolvemethod TISE 



# to run much later
# python rl_train.py -numworkers 8 -study IBMvalencia_q10_seed167_workers8_chan0123_dur2240dt_seg40_sub0.1_delay1 -targetgate NOTCCNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 2240 -numseg 40 -channel 0123 -td3policydelay 1 -evolvemethod TDSE 

