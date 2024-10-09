#!bin/bash -x

if [ -e  submission.sh ]
then
rm submission.sh
fi

chan=12 #012
dur=1120 #1760 #1680 #960 #800
batch=64
numworkers=4
seg=20 #60
sub=0.1 #0.05
ratio=10
#for ratio in 10 #4 #8 12 18 24
#for numgpus in 1 2 3 4
for hid in 400,200,100
do  
    for seed in 167 1 2 #4 5 6 7 8 #167 1 2 3
    do
        #for delay in 1 2 4
	#for batch in 64
	#for chan in 12 03 012 023 123 0123
	for ctrlnoise in 0 1e-3 5e-3 1e-2 2e-2 5e-2 10e-2
        do	
            echo "#!/bin/bash -x" >> submission.sh

            echo "#SBATCH --account=netenesyquma" >> submission.sh
            echo "#SBATCH --nodes=1" >> submission.sh 
            #echo "#SBATCH --ntasks=1" >> submission.sh
            echo "#SBATCH --ntasks-per-node=12" >> submission.sh
            echo "#SBATCH --cpus-per-task="$numworkers >> submission.sh
            echo "#SBATCH --output=/p/project/netenesyquma/nam/drl_gatedesign/data/out.%j" >> submission.sh
            echo "#SBATCH --error=/p/project/netenesyquma/nam/drl_gatedesign/data/err.%j" >> submission.sh

            echo "#SBATCH --time=06:00:00" >> submission.sh
            echo "#SBATCH --partition=batch" >> submission.sh
	    #echo "#SBATCH --partition=booster" >> submission.sh
	    #echo "#SBATCH --gres=gpu:"$numgpus >> submission.sh

            echo eval "$(conda shell.bash hook)" >> submission.sh
            echo module load OpenGL >> submission.sh
            echo conda activate julius >> submission.sh

        
            # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study chan123_dur270_delay1_seg${numseg}_sub${subactionscale} -duration 270 -td3policydelay 1 -numseg $numseg -subactionscale $subactionscale -numworkers 6 -numiter 20000 -seed 167 -channels 123 -targetgate NOTC >> submission.sh
            
            # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study chan23_delay${delay}_dur${duration} -duration $duration -td3policydelay $delay -numworkers 6 -numiter 20000 -seed 167 -targetgate CNOT -channels 23 >> submission.sh
            
            # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study chan13_delay1_dur${duration}_seg${numseg}_sub${subactionscale} -duration $duration -td3policydelay 1 -numseg $numseg -subactionscale $subactionscale -numworkers 6 -numiter 20000 -seed 167 -channels 13 -targetgate HH >> submission.sh
            
            # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study valencia_delay${delay}_dur${duration} -duration $duration -td3policydelay $delay -numworkers 6 -numiter 20000 -seed 167 -targetgate $targetgate -drivestrength 20.47,204.74,15.85,158.54 -anharmonicity ' -310.54,-313.86' -detuning ' -86.66,0' -coupling 2.21  >> submission.sh
            
            # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers 6 -study IBMvalencia_q10_dur${duration}_sub${subactionscale} -targetgate $targetgate -IBMbackend valencia -IBMqubits 10 -subactionscale $subactionscale -numiter 20000 -duration $duration -seed 167 -chpt >> submission.sh
            
            # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers 6 -study IBMvalencia_q10_seed167_chan123_dur${duration}_sub${subactionscale} -targetgate $targetgate -IBMbackend valencia -IBMqubits 10 -subactionscale $subactionscale -numiter 20000 -duration $duration -seed 167 -channels 123 -chpt >> submission.sh
            
            # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers 4 -study IBMvalencia_q10_seed${seed}_sub${subactionscale}_delay${delay}_chan123_dur200_numseg40 -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale $subactionscale -numiter 20000 -duration 200 -numseg 40 -seed $seed -channels 123 -td3policydelay $delay -chpt >> submission.sh

	    # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study workers${numworkers}_IBMvalencia_q10_ratio${UDratio}_seed167_chan03_dur1120dt_seg20_sub0.1 -targetgate XZp90 -IBMbackend valencia -IBMqubits 10 -IBMUDratio $UDratio -subactionscale 0.1 -numiter 20000 -duration 1120 -numseg 20 -seed 167 -channels 03 -evolvemethod TDSE >> submission.sh

	   # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study workers${numworkers}_IBMvalencia_q10_ratio${ratio}_seed${seed}_chan${chan}_dur2240dt_seg${seg}_sub0.1_batch150_delay1 -td3policydelay 1 -batchsize 150 -targetgate NOTCCNOT -IBMbackend valencia -IBMqubits 10 -IBMUDratio $ratio -subactionscale 0.1 -numiter 20000 -duration 2240 -numseg $seg -seed $seed -channels $chan -evolvemethod TDSE >> submission.sh

	   ##### Run to get RL table #####
	   # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_ratio${ratio}_seed${seed}_chan${chan}_TISE_dur1120dt_seg${seg}_sub0.1_batch${batch} -batchsize $batch -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -IBMUDratio $ratio -subactionscale 0.1 -numiter 20000 -duration 1120 -numseg $seg -seed $seed -channels $chan -evolvemethod TISE >> submission.sh

	   # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_ratio${ratio}_seed${seed}_chan${chan}_TISE_dur1120dt_seg${seg}_sub0.1_batch${batch} -batchsize $batch -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -IBMUDratio $ratio -subactionscale 0.1 -numiter 20000 -duration 1120 -numseg $seg -seed $seed -channels $chan -evolvemethod TISE >> submission.sh

	   ## Shorter gatetime
	   # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_ratio${ratio}_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_batch${batch}_avg -batchsize $batch -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -IBMUDratio $ratio -subactionscale $sub -numiter 20000 -duration $dur -numseg $seg -seed $seed -channels $chan -evolvemethod TISE -rewardtype average >> submission.sh

	   # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_ratio${ratio}_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_batch${batch}_delay${delay}_avg -batchsize $batch -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -IBMUDratio $ratio -subactionscale $sub -numiter 20000 -duration $dur -numseg $seg -seed $seed -channels $chan -evolvemethod TISE -rewardtype average -td3policydelay $delay >> submission.sh
       
	   ## Double Hadamard
	   # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_ratio${ratio}_seed${seed}_chan${chan}_dur${dur}dt_seg${seg}_sub${sub}_avg -targetgate HH -IBMbackend valencia -IBMqubits 10 -IBMUDratio $ratio -subactionscale $sub -numiter 20000 -duration $dur -numseg $seg -seed $seed -channels $chan -rewardtype average >> submission.sh		

	   ##### Experimental run #######
	   # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_ratio${ratio}_seed${seed}_chan${chan}_dur1120dt_seg${seg}_sub0.1_batch${batch}_avg -batchsize $batch -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -IBMUDratio $ratio -subactionscale 0.1 -numiter 20000 -duration 1120 -numseg $seg -seed $seed -channels $chan -rewardtype average >> submission.sh

	   # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_ratio${ratio}_seed${seed}_chan${chan}_dur1120dt_seg${seg}_sub0.1_batch${batch} -batchsize $batch -targetgate ZXp90 -IBMbackend valencia -IBMqubits 10 -IBMUDratio $ratio -subactionscale 0.1 -numiter 20000 -duration 1120 -numseg $seg -seed $seed -channels $chan >> submission.sh

	   ###### Larger numseg #########
	   # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_ratio${ratio}_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_batch${batch}_twinq_delay${delay}_avg -batchsize $batch -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -IBMUDratio $ratio -subactionscale $sub -numiter 20000 -duration $dur -numseg $seg -seed $seed -channels $chan -evolvemethod TISE -rewardtype average -td3twinq -td3policydelay $delay >> submission.sh
	   
	   ###### Other gates ########
	   # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_ratio${ratio}_seed${seed}_chan${chan}_dur${dur}dt_seg${seg}_sub${sub}_batch${batch}_twinq_delay${delay}_avg -batchsize $batch -targetgate NOTC -IBMbackend valencia -IBMqubits 10 -IBMUDratio $ratio -subactionscale $sub -numiter 20000 -duration $dur -numseg $seg -seed $seed -channels $chan -rewardtype average -td3twinq -td3policydelay $delay >> submission.sh
	   
	   # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_ratio${ratio}_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_batch${batch}_twinq_delay${delay}_avg -batchsize $batch -targetgate NOTC -IBMbackend valencia -IBMqubits 10 -IBMUDratio $ratio -subactionscale $sub -numiter 20000 -duration $dur -numseg $seg -seed $seed -channels $chan -evolvemethod TISE -rewardtype average -td3twinq -td3policydelay $delay >> submission.sh

	   ###### Control parameters ####
	   # echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_avg_noise${ctrlnoise}_ketctrl -batchsize $batch -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale $sub -numiter 20000 -duration $dur -numseg $seg -seed $seed -channels $chan -evolvemethod TISE -rewardtype average -ctrlnoise $ctrlnoise -rlstate ket_ctrl >> submission.sh

	   echo srun python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -numgpus $numgpus -study juwels_${numworkers}w_${numgpus}gpu_IBMvalencia_q10_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_avg_noise${ctrlnoise}_ketctrl -batchsize $batch -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale $sub -numiter 20000 -duration $dur -numseg $seg -seed $seed -channels $chan -evolvemethod TISE -rewardtype average -ctrlnoise $ctrlnoise -rlstate ket_ctrl >> submission.sh
	   
	   sbatch submission.sh
	   rm submission.sh
        done
    done
done
