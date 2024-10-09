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
  
numworkers=4
chan=2
dur=45
seg=9
sub=0.4
#ctrlnoise=5e-3
# hid="200,400,200"
for hid in 100,200,100
do  
    for seed in 1 2 3 4
    do
        # for ctrlnoise in 2e-2 5e-2 #5e-3 1e-2 #2e-2 
        for rewardtype in average worst
        do	
            #--mem-per-cpu=1GB 
            # echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -study juwels_${numworkers}w_IBMvalencia_q10_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_avg_noise${ctrlnoise}_detune0_ketdetune0_hid${hid} -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale $sub -numiter 20000 -duration $dur -numseg $seg -seed $seed -channels $chan -evolvemethod TISE -rewardtype average -ctrlnoise $ctrlnoise -ctrlnoiseparam detune0 -rlstate ket_detune_0 -hidsizes $hid "&" >> submission.sh
            
            # IX90_4w_dur45dt_seg9_sub0.4_chan2_TISE_avg_hid100,200,100_ratio3_500iter_
            # Single qubit X90 with worst and average
            echo srun --exclusive --ntasks=1 --cpus-per-task=$((numworkers+1)) python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_${numworkers}w_IBMvalencia_q0_ratio3_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_hid${hid}_${rewardtype}_seed${seed} -numtransmon=1 -targetgate=X90 -numworkers=$numworkers -IBMbackend=valencia -IBMqubits=0 -IBMUDratio=3 -seed=$seed -channels=$chan -evolvemethod=TISE -duration=$dur -numseg=$seg -subactionscale=$sub -numiter=500 -rewardtype=${rewardtype} -hidsizes=$hid "&" >> submission.sh
            
            echo >> submission.sh
            echo sleep 60 >> submission.sh
            echo >> submission.sh      
        done
    done
done
echo >> submission.sh
echo wait >> submission.sh
sbatch submission.sh
#rm submission.sh
