#!bin/bash -x
eval "$(conda shell.bash hook)"
conda activate julius

# python rl_inference.py -run 2394 -chpt 4400 -map -concat
# python rl_inference.py -run 7352 -chpt 3000 -map -concat

# 0392_4200
# 2104_4200
# 9109_6000
# 1732_3400
# 5043_2600
# 8639_2400
# 7221_2800

# python rl_inference.py -run 0392 -chpt 4200
# python rl_inference.py -run 2104 -chpt 4200
# python rl_inference.py -run 9109 -chpt 6000
# python rl_inference.py -run 1732 -chpt 3400
# python rl_inference.py -run 5043 -chpt 2600
# python rl_inference.py -run 8639 -chpt 2400
# python rl_inference.py -run 7221 -chpt 2800

for chpt in $(seq 1000 200 4200)
do
    python rl_inference.py -run=6052_3 -chpt=chpt
done