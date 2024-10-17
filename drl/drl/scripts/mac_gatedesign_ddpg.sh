#!bin/bash -x
eval "$(conda shell.bash hook)"
conda activate julius64


### Single gates ###
# python rl_train.py -numworkers 4 -study mac_4w_IBMvalencia_q0_seed1_dur160dt_seg8_sub0.02_avg_hid50,100,50 -numtransmon 1 -targetgate X90 -IBMbackend valencia -IBMqubits 0 -seed 1 -duration 160 -numseg 8 -subactionscale 0.02 -channels 0 -evolvemethod TISE -rewardtype average -hidsizes 50,100,50

# python rl_train.py -numworkers 4 -study mac_4w_IBMvalencia_q1_seed1_dur160dt_seg8_sub0.02_avg_hid50,100,50 -numtransmon 1 -targetgate X90 -IBMbackend valencia -IBMqubits 1 -seed 1 -duration 160 -numseg 8 -subactionscale 0.02 -channels 0 -evolvemethod TISE -rewardtype average -hidsizes 50,100,50

# python rl_train.py -numworkers=4 -study=mac_4w_IBMvalencia_q0_seed1_dur45dt_seg9_sub0.4_hid50,100,50_worst_SCQP -numtransmon 1 -targetgate=X90 -IBMbackend=valencia -IBMqubits=0 -seed 1 -duration=45 -numseg=9 -subactionscale 0.4 -channels=0 -evolvemethod=TISE -rewardtype=worst -worstfidmethod='SCQP' -hidsizes=50,100,50

### Single gates in 2 transmon setting
# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed1_dur45dt_seg9_sub0.1_chan2_TISE_avg_hid100,200,100 -targetgate IX90 -IBMbackend valencia -IBMqubits 10 -seed 1 -duration 45 -numseg 9 -subactionscale=0.1 -channels 2 -evolvemethod TISE -rewardtype average -hidsizes 100,200,100 -numiter=2000

# python rl_train.py -numworkers 4 -study mac_4w_IBMvalencia_q10_seed1_dur45dt_seg9_sub0.4_chan2_TISE_avg_hid100,200,100_ratio5_500iter -targetgate IX90 -IBMbackend valencia -IBMqubits 10 -seed 1 -duration 45 -numseg=9 -subactionscale=0.4 -channels 2 -evolvemethod TISE -rewardtype average -hidsizes 100,200,100 -numiter=500 -evaluationinterval=50 -IBMUDratio=5

# python rl_train.py -numworkers 4 -study mac_4w_IBMvalencia_q10_seed1_dur45dt_seg9_sub0.4_chan2_TISE_avg_hid100,200,100_ratio3_500iter -targetgate IX90 -IBMbackend valencia -IBMqubits 10 -seed 1 -duration 45 -numseg=9 -subactionscale=0.4 -channels 2 -evolvemethod TISE -rewardtype average -hidsizes 100,200,100 -numiter=500 -evaluationinterval=50 -IBMUDratio=3

# for sub in $(seq 0.1 0.1 0.5)
# do
#     # python rl_train.py -numworkers 4 -study mac_4w_IBMvalencia_q10_seed1_dur63dt_seg9_sub${sub}_chan2_TISE_avg_hid100,200,100_ratio5 -targetgate IX90 -IBMbackend valencia -IBMqubits 10 -seed 1 -duration 63 -numseg=9 -subactionscale=$sub -channels 2 -evolvemethod TISE -rewardtype average -hidsizes 100,200,100 -numiter=100 -evaluationinterval=50 -IBMUDratio=5
    
#     python rl_train.py -numworkers 4 -study mac_4w_IBMvalencia_q10_seed1_dur45dt_seg9_sub${sub}_chan2_TISE_avg_hid100,200,100_ratio3 -targetgate IX90 -IBMbackend valencia -IBMqubits 10 -seed 1 -duration 45 -numseg=9 -subactionscale=$sub -channels 2 -evolvemethod TISE -rewardtype average -hidsizes 100,200,100 -numiter=100 -evaluationinterval=50 -IBMUDratio=3
    
# done



### Varying device param ###
# train on noisy environment without context # 
# for noise in 1e-2 5e-2
# do
#     python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed1_chan12_TISE_dur1120dt_seg20_sub0.1_avg_noise${noise}_detune0_hid800,800,800 -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 1 -duration 1120 -numseg 20 -channels 12 -evolvemethod TISE -rewardtype average -ctrlnoise=noise -ctrlnoiseparam detune0 -hidsizes 800,800,800 -numiter 20000 -testcount=20
# done

# python rl_train.py -numworkers=4 -study=5370 -targetgate CNOT -testcount=20 -chpt=30000 -numiter=35000 -rstudy=_normalnoise1e-2 -rctrlnoise=1e-2 -rctrlnoisedist=normal  

# python rl_train.py -numworkers=4 -study=3723 -targetgate CNOT -testcount=20 -chpt=10200 -numiter=15000 -rstudy=_normalnoise1e-4_drift1_seed2 -rctrlnoise=1e-4 -rctrlnoisedist=normal -drift=valencia10_drift_1 -rseed=2
# to change seed, # of random steps


# python rl_train.py -numworkers=4 -study=mac_4w_IBMvalencia_q10_seed1_chan2_TISE_avg_fixedpulse_ent0.99991 -targetgate=ZXp90 -IBMbackend=valencia -IBMqubits=10 -seed=1 -channels=2 -evolvemethod=TISE -rewardtype=average -fixedpulse=../../..-GS_1120dt_20seg_ent0.99991-1

# python rl_train.py -numworkers=4 -study=mac_4w_IBMvalencia_q10_seed1_chan2_TISE_avg_fixedpulse_ent0.99904 -targetgate=ZXp90 -IBMbackend=valencia -IBMqubits=10 -seed=1 -channels=2 -evolvemethod=TISE -rewardtype=average -fixedpulse=../../..-GS_1120dt_20seg_ent0.99904-1

# -IBMUDratio

# python rl_train.py -numworkers=4 -study=mac_4w_IBMvalencia_q10_TISE_avg_drift1_hid800x3_seed1 -targetgate=CNOT -IBMbackend=valencia -IBMqubits=10 -seed=1 -evolvemethod=TISE -rewardtype=average -hidsizes=800,800,800 -numiter=20000 -drift=1 

python rl_train.py -numworkers=4 -study=mac_4w_IBMvalencia_q0_seed1_dur45dt_seg9_sub0.4_hid100,200,100_worst_SCQP -numtransmon=1 -targetgate=X90 -IBMbackend=valencia -IBMqubits=0 -seed=1 -duration=45 -numseg=9 -subactionscale=0.4 -channels=0 -evolvemethod=TISE -rewardtype=worst -worstfidmethod='SCQP-dm-0' -hidsizes=100,200,100