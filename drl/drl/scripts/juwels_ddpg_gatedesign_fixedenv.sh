#!bin/bash -x

if [ -e  submission.sh ]
then
rm submission.sh
fi

echo "#!/bin/bash -x" >> submission.sh
echo "#SBATCH --job-name=hardware" >> submission.sh
echo "#SBATCH --account=netenesyquma" >> submission.sh
echo "#SBATCH --nodes=1" >> submission.sh 
echo "#SBATCH --ntasks=8" >> submission.sh 
echo "#SBATCH --ntasks-per-node=8" >> submission.sh
echo "#SBATCH --output=/p/project/netenesyquma/nam/drl_gatedesign/data/out.%j" >> submission.sh
echo "#SBATCH --error=/p/project/netenesyquma/nam/drl_gatedesign/data/err.%j" >> submission.sh

echo "#SBATCH --time=24:00:00" >> submission.sh
echo "#SBATCH --partition=batch" >> submission.sh

echo eval "$(conda shell.bash hook)" >> submission.sh
echo module load OpenGL >> submission.sh
echo conda activate julius >> submission.sh
  
  
### SINGLE TRANSMON ###
numworkers=4
chan=0
dur=45
seg=9
sub=0.4
#ctrlnoise=5e-3
# hid="200,400,200"
rewardtype=average
for hid in 100,200,100 50,100,50
do  
    for seed in 1 2 3 4
    do
        # for ctrlnoise in 2e-2 5e-2 #5e-3 1e-2 #2e-2 
        for method in '' #SCQP-dm-0 SLSQP-ket-3  #SCQP
        do	
            #--mem-per-cpu=1GB 
            # echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_avg_noise${ctrlnoise}_detune0_ketdetune0_hid${hid} -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale $sub -numiter 20000 -duration $dur -numseg $seg -seed $seed -channels $chan -evolvemethod TISE -rewardtype average -ctrlnoise $ctrlnoise -ctrlnoiseparam detune0 -rlstate ket_detune_0 -hidsizes $hid "&" >> submission.sh
            
            # IX90_4w_dur45dt_seg9_sub0.4_chan2_TISE_avg_hid100,200,100_ratio3_500iter_
            # Single qubit X90 with worst and average
            echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_${numworkers}w_IBMvalencia_q0_ratio3_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_hid${hid}_${rewardtype}_${method}_seed${seed} -numtransmon=1 -targetgate=X90 -numworkers=$numworkers -IBMbackend=valencia -IBMqubits=0 -IBMUDratio=3 -seed=$seed -channels=$chan -evolvemethod=TISE -duration=$dur -numseg=$seg -subactionscale=$sub -numiter=5000 -rewardtype=$rewardtype -hidsizes=$hid -worstfidmethod=$method "&" >> submission.sh
            
            echo >> submission.sh
            echo sleep 60 >> submission.sh
            echo >> submission.sh      
        done
    done
done


# ### TWO TRANSMONS ###
# numworkers=4
# chan=1
# dur=1120
# seg=20
# sub=0.1
# rewardtype=average
# for _ in 100
# do  
#     for seed in 1 2 3 4
#     do
#         for _ in 100
#         do	            
#             echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_${numworkers}w_IBMvalencia_q10_ratio10_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_${rewardtype}_seed${seed} -targetgate=ZXp90 -numworkers=$numworkers -IBMbackend=valencia -IBMqubits=10 -IBMUDratio=10 -seed=$seed -channels=$chan -evolvemethod=TISE -duration=$dur -numseg=$seg -subactionscale=$sub -numiter=10000 -rewardtype=$rewardtype "&" >> submission.sh
            
#             echo >> submission.sh
#             echo sleep 60 >> submission.sh
#             echo >> submission.sh      
#         done
#     done
# done

# ### TWO TRANSMONS - Shorter gatetime ###
# numworkers=4
# chan=12
# dur=640
# seg=20
# sub=0.15
# targetgate=ZXp90
# rewardtype=average
# for _ in 100
# do  
#     for seed in 1 2 3 4 5 6 7 8
#     do
#         for _ in 100
#         do	            
#             echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_${numworkers}w_IBMvalencia_q10_ratio10_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_${rewardtype}_seed${seed} -targetgate=$targetgate -numworkers=$numworkers -IBMbackend=valencia -IBMqubits=10 -IBMUDratio=10 -seed=$seed -channels=$chan -evolvemethod=TISE -duration=$dur -numseg=$seg -subactionscale=$sub -numiter=10000 -rewardtype=$rewardtype "&" >> submission.sh
            
#             echo >> submission.sh
#             echo sleep 60 >> submission.sh
#             echo >> submission.sh      
#         done
#     done
# done

# ### TWO TRANSMONS - 3 channels + TD3 ###
# numworkers=4
# chan=012
# dur=800
# seg=20
# sub=0.15
# targetgate=CNOT
# rewardtype=average
# for delay in _ # 1 2
# do  
#     for seed in 1 2 3 4 5 6 7 8
#     do
#         for _ in 100
#         do	            
#             # echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_${numworkers}w_IBMvalencia_q10_ratio10_chan${chan}_TDSE_dur${dur}dt_seg${seg}_sub${sub}_${rewardtype}_twinq_delay${delay}_seed${seed} -targetgate=$targetgate -numworkers=$numworkers -IBMbackend=valencia -IBMqubits=10 -IBMUDratio=10 -seed=$seed -channels=$chan -evolvemethod=TDSE -duration=$dur -numseg=$seg -subactionscale=$sub -numiter=10000 -rewardtype=$rewardtype -td3twinq -td3policydelay=$delay  "&" >> submission.sh
            
#             # DDPG for chan012
#             echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_${numworkers}w_IBMvalencia_q10_ratio10_chan${chan}_TDSE_dur${dur}dt_seg${seg}_sub${sub}_${rewardtype}_seed${seed} -targetgate=$targetgate -numworkers=$numworkers -IBMbackend=valencia -IBMqubits=10 -IBMUDratio=10 -seed=$seed -channels=$chan -evolvemethod=TDSE -duration=$dur -numseg=$seg -subactionscale=$sub -numiter=10000 -rewardtype=$rewardtype  "&" >> submission.sh
            
#             echo >> submission.sh
#             echo sleep 60 >> submission.sh
#             echo >> submission.sh      
#         done
#     done
# done


echo >> submission.sh
echo wait >> submission.sh
sbatch submission.sh
#rm submission.sh
