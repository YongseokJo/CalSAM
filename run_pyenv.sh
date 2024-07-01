#!/bin/bash
#SBATCH --job-name=ILI_SAM   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=g.kerex@gmail.com     # Where to send mail	
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
###SBATCH -p gpu --gpus=<gpu_type>:<N> -c <M>
###SBATCH -p gpu --gpus=v100s-32gb:1 -c 16
#SBATCH -p cmbas -c 1
###SBATCH -p gpu --gpus=1 -c 12

pwd; hostname; date

module load gcc
module load python3

pyenv install 3.6.4

date
