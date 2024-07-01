#!/bin/bash
#SBATCH --job-name=OC_SAM   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=g.kerex@gmail.com     # Where to send mail	
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
###SBATCH -p gpu --gpus=<gpu_type>:<N> -c <M>
###SBATCH -p gpu --gpus=v100s-32gb:1 -c 16
#SBATCH -p cca -c 50
###SBATCH -A cca

pwd; hostname; date

module add python3

./one_click_SAM.py -s TNG300 -n smz_smgf  > stdout 2>stderr
~/pyenv/venv/bin/python3  ./sam_to_numpy.py -s TNG300 -n smz_smgf 

date