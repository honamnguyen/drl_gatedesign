#!bin/bash -x
eval "$(conda shell.bash hook)"
conda activate julius

# python rl_train.py -numworkers 4 -study valencia_test -targetgate ZXp90 -drivestrength 20.47,204.74,15.85,158.54 -anharmonicity ' -310.54,-313.86' -detuning ' -86.66,0' -coupling 2.21

# python rl_train.py -numworkers 6 -study IBMvalencia_q10_seed167_sub0.15 -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.15 -seed 167

# python rl_train.py -numworkers 4 -study IBMvalencia_q10_seed167_sub0.075 -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.075 -seed 167

# python rl_train.py -numworkers 6 -study valencia_sub0.1 -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -chpt

# python rl_train.py -numworkers 6 -study IBMvalencia_q10_seed167_workers6_chan3_dur15_seg10_sub0.4_avg -targetgate IX90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.4 -seed 167 -duration 15 -numseg 10 -rewardtype average -channel 3

# python rl_train.py -numworkers 6 -study IBMvalencia_q10_seed167_workers6_chan2_dur128dt_seg8_sub0.4 -targetgate IX90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.4 -seed 167 -duration 128 -numseg 8 -channel 2 -evolvemethod TISE

# python rl_train.py -numworkers 6 -study IBMvalencia_q10_seed167_workers6_chan2_dur128dt_seg8_sub0.4 -targetgate X90I -IBMbackend valencia -IBMqubits 10 -subactionscale 0.4 -seed 167 -duration 128 -numseg 8 -channel 0 -evolvemethod TDSE
 
python rl_train.py -numworkers 4 -study IBMvalencia_q10_seed167_workers4_chan012_dur1120dt_seg20_sub0.1 -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -duration 1120 -numseg 20 -channel 012 -evolvemethod TDSE

# python rl_train.py -numworkers 6 -study IBMvalencia_q10_seed167_workers6_chan2_dur128dt_seg8_sub0.4 -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 

# for numworkers in 0 4
# do
#     for hidsizes in 800 800,400,200
#     do
#         python rl_train.py -numworkers ${numworkers} -study SeedRef_workers${numworkers}_hid${hidsizes}_seed167_afterinit_chpt -targetgate X90 -numtransmon 1 -numseg 5 -fidthreshold 0.9999 -worstfidmethod SCQP-dm-0 -channels 1 -duration 10 -IBMbackend valencia -IBMqubits 1 -subactionscale 0.4 -seed 167 -replayinitial 3000 -hidsizes $hidsizes -numiter 0

#         for s in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
#         do
#             python rl_train.py -numworkers ${numworkers} -study Seed_workers${numworkers}_hid${hidsizes}_seed167_afterinit_chpt -targetgate X90 -numtransmon 1 -numseg 5 -fidthreshold 0.9999 -worstfidmethod SCQP-dm-0 -channels 1 -duration 10 -IBMbackend valencia -IBMqubits 1 -subactionscale 0.4 -seed 167 -replayinitial 3000 -hidsizes $hidsizes -numiter 6 -chptrun Ref_workers${numworkers}_hid${hidsizes} -chpt 000
#         done
#     done
# done

# for s in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
# do
#     python rl_train.py -numworkers 4 -study Seed_workers4_hid800,400,200_seed167_afterinit -targetgate X90 -numtransmon 1 -numseg 5 -fidthreshold 0.9999 -worstfidmethod SCQP-dm-0 -channels 1 -duration 10 -IBMbackend valencia -IBMqubits 1 -subactionscale 0.4 -seed 167 -replayinitial 3000 -hidsizes 800,400,200 -numiter 6
# done

# for s in 1 2 3 4 5 6 7 8 9 10
# do
#     python rl_train.py -numworkers 4 -study Seed_workers4_hid800,400,200_seed167_afterinit -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 167 -replayinitial 3000 -numiter 8
# done