#!/bin/bash -x
eval "$(conda shell.bash hook)"
conda activate julius64
cd qtool
pip install -e .
cd ../drl
pip install -e .
cd ../gym
pip install -e .
