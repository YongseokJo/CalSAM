#!/bin/bash
#SBATCH --job-name=ILI_SAM   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=g.kerex@gmail.com     # Where to send mail	
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
###SBATCH -p gpu --gpus=<gpu_type>:<N> -c <M>
###SBATCH -p gpu --gpus=v100s-32gb:1 -c 16
###SBATCH -p cmbas -c 36
#SBATCH -p gpu --gpus=1 -c 12
#SBATCH -A cca

pwd; hostname; date

module load gcc
module add modules/2.0-20220630
moulde add  cuda
moulde add  cudnn
module load python3


~/pyenv/venv/bin/python3 ./ili.py -b smz smgf -n smz_smgf -N 50  > stdout 2>stderr

date
