#!/bin/bash
#SBATCH --job-name=smz_gf   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=g.kerex@gmail.com     # Where to send mail	
#SBATCH --time=120:00:00               # Time limit hrs:min:sec
###SBATCH -p gpu --gpus=<gpu_type>:<N> -c <M>
###SBATCH -p gpu --gpus=v100s-32gb:1 -c 16
###SBATCH -p gen -c 1
#SBATCH -p gen -c 1 
###-A cca

pwd; hostname; date

module load gcc
#module add cuda
#module add cudnn
module add python3


~/pyenv/venv/bin/python3 ./wrapper.py -s TNG300 -b smz smgf -n smz_smgf -N 50 > stdout_wrapper 2>stderr_wrapper

date
