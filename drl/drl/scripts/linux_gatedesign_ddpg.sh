#!bin/bash -x
eval "$(conda shell.bash hook)"
conda activate julius


### Single gates ###
# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q0_seed1_dur160dt_seg8_sub0.02_avg_hid50,100,50 -numtransmon 1 -targetgate X90 -IBMbackend valencia -IBMqubits 0 -seed 1 -duration 160 -numseg 8 -subactionscale 0.02 -channels 0 -evolvemethod TISE -rewardtype average -hidsizes 50,100,50

# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q1_seed1_dur160dt_seg8_sub0.02_avg_hid50,100,50 -numtransmon 1 -targetgate X90 -IBMbackend valencia -IBMqubits 1 -seed 1 -duration 160 -numseg 8 -subactionscale 0.02 -channels 0 -evolvemethod TISE -rewardtype average -hidsizes 50,100,50

### Sinlge gates in 2 transmon setting
# python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed1_dur45dt_seg9_sub0.1_chan2_TISE_avg_hid100,200,100 -targetgate IX90 -IBMbackend valencia -IBMqubits 10 -seed 1 -duration 45 -numseg 9 -subactionscale=0.1 -channels 2 -evolvemethod TISE -rewardtype average -hidsizes 100,200,100 -numiter=2000

# for sub in 0.4 0.2
# do
#     python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed1_dur90dt_seg9_sub${sub}_chan2_TISE_avg_hid100,200,100 -targetgate IX90 -IBMbackend valencia -IBMqubits 10 -seed 1 -duration 90 -numseg=9 -subactionscale=$sub -channels 2 -evolvemethod TISE -rewardtype average -hidsizes 100,200,100 -numiter=1000
# done


### Varying device param ###
# train on noisy environment without context # 
# for noise in 5e-2 #1e-2 
# do
    # no context, uniform noise
    # python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed1_chan12_TISE_dur1120dt_seg20_sub0.1_avg_uniformnoise${noise}_all_hid800,800,800 -targetgate=CNOT -IBMbackend=valencia -IBMqubits 10 -subactionscale 0.1 -seed 1 -duration 1120 -numseg 20 -channels 12 -evolvemethod TISE -rewardtype average -ctrlnoise=$noise -ctrlnoiseparam all -hidsizes 800,800,800 -numiter 20000 -testcount=20
    
    # python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_q10_seed1_TISE_avg_uniformnoise${noise}_all_ketall_hid800x3 -targetgate=CNOT -IBMbackend=valencia -IBMqubits=10 -seed=1 -evolvemethod=TISE -rewardtype=average -ctrlnoise=${noise} -ctrlnoiseparam=all -rlstate=ket_all -hidsizes=800,800,800 -numiter=40000 -testcount=20 -chpt=30000
    
    # python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed1_TISE_avg_uniformnoise${noise}_detune0_ketdetune0_hid800x3 -targetgate=CNOT -IBMbackend=valencia -IBMqubits=10 -seed=1 -evolvemethod=TISE -rewardtype=average -ctrlnoise=$noise -ctrlnoiseparam=detune0 -rlstate=ket_detune0 -hidsizes=800,800,800 -numiter=20000 -testcount=20 
    
    # python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed1_chan12_TISE_dur1120dt_seg20_sub0.1_avg_uniformnoise${noise}_detune0_anharm_hid800,800,800 -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 1 -duration 1120 -numseg 20 -channels 12 -evolvemethod TISE -rewardtype average -ctrlnoise=$noise -ctrlnoiseparam detune0_anharm -hidsizes 800,800,800 -numiter 20000 -testcount=20
    
    # python rl_train.py -numworkers 4 -study linux_4w_IBMvalencia_q10_seed1_chan12_TISE_dur1120dt_seg20_sub0.1_avg_uniformnoise${noise}_detune0_ketdetune0 -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale 0.1 -seed 1 -duration 1120 -numseg 20 -channels 12 -evolvemethod TISE -rewardtype average -ctrlnoise=$noise -ctrlnoiseparam detune0 -numiter 20000 -testcount=20 -rlstate ket_detune_0
# done


        
# python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_q10_seed1_TISE_avg_uniformnoise5e-2_all_ketall_hid800x3_seg28 -targetgate=CNOT -IBMbackend=valencia -IBMqubits=10 -seed=1 -evolvemethod=TISE -rewardtype=average -ctrlnoise=5e-2 -ctrlnoiseparam=all -rlstate=ket_all -hidsizes=800,800,800 -numiter=40000 -testcount=20 -numseg=28

# python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_q10_seed1_TISE_avg_uniformnoise5e-2_all_hid800x3_seg28 -targetgate=CNOT -IBMbackend=valencia -IBMqubits=10 -seed=1 -evolvemethod=TISE -rewardtype=average -ctrlnoise=5e-2 -ctrlnoiseparam=all -rlstate=ket -hidsizes=800,800,800 -numiter=40000 -testcount=20 -numseg=28

# python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_q10_seed1_TISE_avg_uniformnoise2e-2_all_hid800x3_seg28 -targetgate=CNOT -IBMbackend=valencia -IBMqubits=10 -seed=1 -evolvemethod=TISE -rewardtype=average -ctrlnoise=5e-2 -ctrlnoiseparam=all -rlstate=ket -hidsizes=800,800,800 -numiter=40000 -testcount=20 -numseg=28

### Restart to increase ctrlnoise ###
# python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_q10_seed1_TISE_avg_normalnoise2e-3_all_ketall -targetgate=CNOT -IBMbackend=valencia -IBMqubits=10 -seed=1 -evolvemethod=TISE -rewardtype=average -ctrlnoise=2e-3 -ctrlnoisedist=normal -ctrlnoiseparam=all -rlstate=ket_all -numiter=5000 -testcount=20

# python rl_train.py -numworkers=4 -study=3385 -targetgate=CNOT -testcount=20 -chpt=10000 -numiter=15000 -rstudy=_normalnoise1e-2 -rctrlnoise=1e-2 -rctrlnoisedist=normal


### Investigate ability to learn at 1.015x device param ###
# python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_all1.015_q10_seed1_TISE_avg -targetgate=CNOT -IBMbackend=valencia_all1.015 -IBMqubits=10 -seed=1 -evolvemethod=TISE -rewardtype=average -numiter=5000 -testcount=20

# python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_all1.015_q10_seed1_TISE_avg_uniformnoise5e-3_all_ketall_hid800x3 -targetgate=CNOT -IBMbackend=valencia_all1.015 -IBMqubits=10 -seed=1 -evolvemethod=TISE -rewardtype=average -ctrlnoise=5e-3 -ctrlnoisedist=uniform -ctrlnoiseparam=all -rlstate=ket_all -hidsizes=800,800,800 -numiter=20000 -testcount=20

# python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_freq1.015_q10_seed1_TISE_avg -targetgate=CNOT -IBMbackend=valencia_freq1.015 -IBMqubits=10 -seed=1 -evolvemethod=TISE -rewardtype=average -numiter=5000 -testcount=20


### Learn on top of fixed CR pulse ###
# for 
# do
#     python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_q10_seed1_chan2_TISE_avg_fixedpulse_ent0.99991 -targetgate=ZXp90 -IBMbackend=valencia -IBMqubits=10 -seed=1 -channels=2 -evolvemethod=TISE -rewardtype=average -fixedpulse=../../..-GS_1120dt_20seg_ent0.99991-1 -numiter=200
# done

 
# python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_q10_seed1_chan2_TISE_avg_fixedpulse_ent0.99605_ctrlfid0.97639 -targetgate=ZXp90 -IBMbackend=valencia -IBMqubits=10 -seed=1 -channels=2 -evolvemethod=TISE -rewardtype=average -fixedpulse=../../..-GS_1120dt_20seg_ent0.99605_ctrlfid0.97639-1 -numiter=500

# python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_q10_seed1_chan02_TDSE_avg_fixedpulse_ent0.99605_ctrlfid0.97639 -targetgate=ZXp90 -IBMbackend=valencia -IBMqubits=10 -seed=1 -channels=02 -evolvemethod=TDSE -rewardtype=average -fixedpulse=../../..-GS_1120dt_20seg_ent0.99605_ctrlfid0.97639-1 -numiter=500


### Everystep noise - robustness ###
# python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_q10_seed1_TISE_avg_normalnoise3e-2_all_everystep_hid800x3_seg28 -targetgate=CNOT -IBMbackend=valencia -IBMqubits=10 -seed=1 -evolvemethod=TISE -rewardtype=average -ctrlnoisedist=normal -ctrlnoise=3e-2 -ctrlupdatefreq=everystep -ctrlnoiseparam=all -rlstate=ket -hidsizes=800,800,800 -numiter=40000 -testcount=20 -numseg=28


# python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_q10_seed167_TISE_avg_hid800x3_seg28 -targetgate=CNOT -IBMbackend=valencia -IBMqubits=10 -seed=167 -evolvemethod=TISE -rewardtype=average -hidsizes=800,800,800 -numiter=10000 -numseg=28

#for seed in 1 2
#do
#    python rl_train.py -numworkers=4 -study=linux_4w_IBMvalencia_q10_TISE_avg_batch150_seed$seed -targetgate=ZXp90 -IBMbackend=valencia -IBMqubits=10 -seed=$seed -evolvemethod=TISE -rewardtype=average -batchsize=150 -numiter=4000
#done

python rl_train.py -numworkers=4 -study=3723 -targetgate CNOT -testcount=20 -chpt=10200 -numiter=15000 -rstudy=_noiseless_drift1_seed3 -rctrlnoise=0 -rctrlnoisedist=normal -drift=valencia10_drift_1 -rseed=3
