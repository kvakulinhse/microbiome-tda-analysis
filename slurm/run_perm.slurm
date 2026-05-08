#!/bin/bash
#SBATCH --job-name=perm
#SBATCH --output=perm.out
#SBATCH --error=perm.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=test

module purge
module load Python/Anaconda_v10.2019
cd ~/tda_project
python perm_slurm.py
