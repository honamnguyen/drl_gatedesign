	    #echo "#SBATCH --partition=booster" >> submission.sh
#!bin/bash -x
#!bin/bash -x

if [ -e  submission.sh ]
then
rm submission.sh
fi

numworkers=4
numgpus=1
echo "#!/bin/bash -x" >> submission.sh
echo "#SBATCH --account=netenesyquma" >> submission.sh
echo "#SBATCH --nodes=2" >> submission.sh 
echo "#SBATCH --ntasks=8" >> submission.sh 
echo "#SBATCH --ntasks-per-node=4" >> submission.sh
echo "#SBATCH --cpus-per-task="${numworkers} >> submission.sh
echo "#SBATCH --output=/p/project/netenesyquma/nam/drl_gatedesign/data/out.%j" >> submission.sh
echo "#SBATCH --error=/p/project/netenesyquma/nam/drl_gatedesign/data/err.%j" >> submission.sh
echo "#SBATCH --time=24:00:00" >> submission.sh
echo "#SBATCH --partition=booster" >> submission.sh
echo "#SBATCH --gres=gpu:4" >> submission.sh    


echo eval "$(conda shell.bash hook)" >> submission.sh
echo module load OpenGL >> submission.sh
echo conda activate julius >> submission.sh
            
chan=12
dur=1120
batch=64
seg=20
sub=0.1
ratio=10
#ctrlnoise=5e-3
for hid in 800,400,200 400,200,100
do  
    for seed in 167 1 #4 5 6 7 8 #167 1 2 3
    do
	for ctrlnoise in 5e-3 1e-2
        do	
            echo srun --gres=gpu:$numgpu python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -numworkers $numworkers -numgpus $numgpus -study juwels_${numworkers}w_${numgpus}gpu_IBMvalencia_q10_seed${seed}_chan${chan}_TISE_dur${dur}dt_seg${seg}_sub${sub}_avg_noise${ctrlnoise}_ketctrl -batchsize $batch -targetgate CNOT -IBMbackend valencia -IBMqubits 10 -subactionscale $sub -numiter 20000 -duration $dur -numseg $seg -seed $seed -channels $chan -evolvemethod TISE -rewardtype average -ctrlnoise $ctrlnoise -rlstate ket_ctrl >> submission.sh	   
        done
    done
done
sbatch submission.sh
rm submission.sh
