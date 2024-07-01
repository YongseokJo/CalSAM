#!/bin/bash
#SBATCH --job-name=p_fzzgf   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=g.kerex@gmail.com     # Where to send mail	
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
###SBATCH -p gpu --gpus=<gpu_type>:<N> -c <M>
###SBATCH -p gpu --gpus=v100s-32gb:1 -c 16
###SBATCH -p cmbas -c 36
#SBATCH -p cca -c 8
#SBATCH -A cca

pwd; hostname; date

module load gcc
module add modules/2.0-20220630
moulde add  cuda
moulde add  cudnn
module load python3


~/pyenv/venv/bin/python3 ./plot_with_sampling.py \
        -d fz_and_all_1000 \
        -ob1 smf smz -ob2 smf smz smgf -n1 20 -n2 11 -s\
        > stdout_plot1 2>stderr_plot1

        #-d all_and_zgf_1000 \

date
