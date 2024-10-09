#!bin/bash -x

if [ -e  submission.sh ]
then
rm submission.sh
fi

numworkers=4
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
            
chan=12
dur=640
seg=20
# sub=0.2
# for seed in 167 1 #2 3 #2
for seed in 2 3 #2
do  
    for delay in -1 #1 2 3 4
    # for dur in 1280 1440
    do
        for sub in 0.15 0.2 0.25 0.3
        do
            # standard
            # echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_${numworkers}w_IBMvalencia_q10_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_twinq_delay${delay}_avg -targetgate=ZXp90 -numworkers=$numworkers -IBMbackend=valencia -IBMqubits=10 -IBMUDratio=10 -seed=$seed -channels=$chan -evolvemethod=TISE -duration=$dur -numseg=$seg -subactionscale=$sub -numiter=20000 -td3twinq -td3policydelay=$delay -rewardtype=average "&" >> submission.sh
            
            # change ratio 
            # echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_${numworkers}w_IBMvalencia_q10_ratio5_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_twinq_delay${delay}_avg -targetgate=ZXp90 -numworkers=$numworkers -IBMbackend=valencia -IBMqubits=10 -IBMUDratio=5 -seed=$seed -channels=$chan -evolvemethod=TISE -duration=$dur -numseg=$seg -subactionscale=$sub -numiter=20000 -td3twinq -td3policydelay=$delay -rewardtype=average "&" >> submission.sh
            
            # ddpg
            # echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_${numworkers}w_IBMvalencia_q10_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_avg -targetgate=ZXp90 -numworkers=$numworkers -IBMbackend=valencia -IBMqubits=10 -IBMUDratio=10 -seed=$seed -channels=$chan -evolvemethod=TISE -duration=$dur -numseg=$seg -subactionscale=$sub -numiter=20000 -rewardtype=average "&" >> submission.sh
            
            # ddpg change ratio
            echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_${numworkers}w_IBMvalencia_q10_ratio5_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_avg -targetgate=ZXp90 -numworkers=$numworkers -IBMbackend=valencia -IBMqubits=10 -IBMUDratio=5 -seed=$seed -channels=$chan -evolvemethod=TISE -duration=$dur -numseg=$seg -subactionscale=$sub -numiter=20000 -rewardtype=average "&" >> submission.sh
            
            # echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_${numworkers}w_IBMvalencia_q10_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_avg -targetgate CNOT -numworkers $numworkers -IBMbackend valencia -IBMqubits 10 -seed $seed -channels $chan -evolvemethod TISE -duration $dur -numseg $seg -subactionscale $sub -numiter 20000 -rewardtype average "&" >> submission.sh
            
            echo >> submission.sh
            echo sleep 40 >> submission.sh
            echo >> submission.sh  
        done
    done
done
echo >> submission.sh
echo wait >> submission.sh
sbatch submission.sh
#rm submission.sh
