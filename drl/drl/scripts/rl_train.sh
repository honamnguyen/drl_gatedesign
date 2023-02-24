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
# conda activate julius

# python rl_train.py -numworkers 4 -study IBMvalencia_q10_seed167_workers4_chan012_dur1120dt_seg20_sub0.1 -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channel 012 -evolvemethod TDSE

# python rl_train.py -numworkers 4 -study IBMvalencia_q10_seed167_workers4_chan012_dur1120dt_seg20_sub0.1_avg -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channel 012 -evolvemethod TDSE -rewardtype average

# python rl_train.py -numworkers 4 -study workers4_IBMvalencia_q10_seed167_chan12_TISE_dur1120dt_seg20_sub0.1_avg -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channel 12 -evolvemethod TISE -rewardtype average

# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed167_chan12_TISE_dur1120dt_seg20_sub0.1_avg -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channel 12 -evolvemethod TISE -rewardtype average

### experiments ###
# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed167_chan012_dur1120dt_seg20_sub0.1_avg -targetgate NOTC -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channel 012 -rewardtype average ## try NOTC, didn't work bleh

# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed167_chan12_TISE_dur800dt_seg20_sub0.2_avg -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.2 -seed 167 -duration 800 -numseg 20 -channel 12 -evolvemethod TISE -rewardtype average ## 177ns which requires avg_amp=0.6, it works

# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed167_chan12_TISE_dur800dt_seg40_sub0.1_avg -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 800 -numseg 40 -channel 12 -evolvemethod TISE -rewardtype average ## 177ns but 40seg so 0.1sub, blow up Q

# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed167_chan12_TISE_dur800dt_seg40_sub0.1_delay1_avg -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 800 -numseg 40 -channel 12 -evolvemethod TISE -rewardtype average -td3policydelay 1 ## 177ns but 40seg so 0.1sub, td3 but not learnng quite well

# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed167_chan12_TISE_dur800dt_seg40_sub0.1_twinq_delay4_avg -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 800 -numseg 40 -channel 12 -evolvemethod TISE -rewardtype average -td3policydelay 4 -td3twinq ## 177ns but 40seg so 0.1sub, remove smoothing stuff, learning but got to 1.8

# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed167_chan12_TISE_dur800dt_seg40_sub0.2_twinq_delay1_avg -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.2 -seed 167 -duration 800 -numseg 40 -channel 12 -evolvemethod TISE -rewardtype average -td3policydelay 1 -td3twinq ## 177ns but 40seg so 0.2sub

# python rl_train.py -numworkers 4 -study IBMvalencia_q10_seed167_workers4_chan012_dur1120dt_seg20_sub0.1_batch150_delay1 -targetgate NOTCCNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channel 012 -batchsize 150 -td3policydelay 1 -evolvemethod TDSE 

# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed167_chan12_TISE_dur1120dt_seg20_sub0.1_avg_noise5e-3_ketctrl -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channels 12 -evolvemethod TISE -rewardtype average -ctrlnoise 5e-3 -rlstate ket_ctrl # Rerun CNOT with ket_ctrl and ctrl_noise

# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed167_chan12_TISE_dur1120dt_seg20_sub0.1_avg_hid200,100,50 -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channels 12 -evolvemethod TISE -rewardtype average -hidsizes 200,100,50 -numiter 20000 # test with smaller networks

# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed167_chan12_TISE_dur1120dt_seg20_sub0.1_avg_noise1e-2_ketctrl_hid800,800,800 -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channels 12 -evolvemethod TISE -rewardtype average -ctrlnoise 1e-2 -rlstate ket_ctrl -hidsizes 800,800,800 -numiter 20000 # Rerun CNOT with ket_ctrl and ctrl_noise, larger network

# python rl_train.py -study linux_4w_IBMvalencia_q10_seed167_chan12_TISE_dur1440dt_seg20_sub0.1_avg_twinq_delay2 -targetgate NOTC -numworkers 4 -IBMbackend valencia -IBMqubits 10 -seed 167 -channels 12 -evolvemethod TISE -duration 1440 -numseg 20 -subactionscale 0.1 -rewardtype average -td3twinq -td3policydelay 2 ## try NOTC again


########### ON MAC ###########
conda activate julius64

# python rl_train.py -numworkers 8 -study workers8_IBMvalencia_q10_seed2_chan12_TISE_dur1120dt_seg20_sub0.1_batch150 -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 2 -duration 1120 -numseg 20 -channel 12 -batchsize 150 -evolvemethod TISE 

# python rl_train.py -numworkers 4 -study workers4_IBMvalencia_q10_seed2_chan12_TISE_dur1120dt_seg20_sub0.1_batch150 -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 2 -duration 1120 -numseg 20 -channel 12 -batchsize 150 -evolvemethod TISE 

# python rl_train.py -numworkers 4 -study mac_4w_IBMvalencia_q10_ratio2_seed167_chan12_TISE_dur1120dt_seg40_sub0.05_twinq_delay2_avg -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -IBMUDratio 2.5 -subactionscale 0.05 -seed 167 -duration 1120 -numseg 40 -channel 12 -evolvemethod TISE -rewardtype average -td3policydelay 2 -td3twinq # Get CNOT to combine to get NOTC, try ratio 2.5 (i.e. max_diff for Hadamard is 0.02)

# python rl_train.py -numworkers 4 -study mac_4w_IBMvalencia_q10_ratio5_seed167_chan12_TISE_dur1120dt_seg40_sub0.05_twinq_delay2_avg -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -IBMUDratio 5 -subactionscale 0.05 -seed 167 -duration 1120 -numseg 40 -channel 12 -evolvemethod TISE -rewardtype average -td3policydelay 2 -td3twinq # Get CNOT to combine to get NOTC, try ratio 5 (i.e. max_diff for Hadamard is 0.01)

python rl_train.py -numworkers 4 -study mac_4w_IBMvalencia_q10_seed167_chan12_TISE_dur1120dt_seg20_sub0.1_avg_noise1e-2_drive_ketdrive -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channels 12 -evolvemethod TISE -rewardtype average -ctrlnoise 1e-2 -ctrlnoiseparam drive -rlstate ket_drive # Rerun CNOT with rl_state=ket_drive and ctrl_noise for drive


# to run much later
# python rl_train.py -numworkers 8 -study IBMvalencia_q10_seed167_workers8_chan0123_dur2240dt_seg40_sub0.1_delay1 -targetgate NOTCCNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 2240 -numseg 40 -channel 0123 -td3policydelay 1 -evolvemethod TDSE 

