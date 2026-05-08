#!/bin/bash
#SBATCH --job-name=perm_nobal
#SBATCH --output=perm_nobal_%j.out
#SBATCH --error=perm_nobal_%j.err
#SBATCH --time=01:30:00

module purge
module load Python/Anaconda_v10.2019
cd ~/tda_project
python perm_clr_no_balance.py
