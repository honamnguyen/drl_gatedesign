#!bin/bash -x
eval "$(conda shell.bash hook)"
conda activate julius

for seed in 1 2 3
do
    for subactionscale in 0.04 0.06
    do
        for delay in 1 2
      	do
            python copy_chpt.py -study IBMvalencia_q10_seed${seed}_sub${subactionscale}_delay${delay}_chan123_dur200_numseg40 -targetgate CNOT
        done
    done
done
