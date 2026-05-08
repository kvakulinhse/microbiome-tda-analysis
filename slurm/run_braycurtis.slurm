#!/bin/bash
#SBATCH --job-name=bray
#SBATCH --output=bray_%j.out
#SBATCH --error=bray_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=test

module purge
module load Python/Anaconda_v10.2019
cd ~/tda_project
python tda_braycurtis.py
