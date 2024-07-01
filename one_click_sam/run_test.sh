#!/bin/bash
#SBATCH --job-name=test3   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=g.kerex@gmail.com     # Where to send mail	
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
###SBATCH -p gpu --gpus=<gpu_type>:<N> -c <M>
###SBATCH -p gpu --gpus=v100s-32gb:1 -c 16
#SBATCH -p cca -c 1

pwd; hostname; date

module add python3

#~/pyenv/venv/bin/python3  ./sam_to_numpy.py
./one_click_v2_test.py >test/stdout 2> test/stderr

date
