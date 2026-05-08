#!/bin/bash
#SBATCH --job-name=clr_aitch
#SBATCH --output=clr_%j.out
#SBATCH --error=clr_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=normal

module purge
module load Python/Anaconda_v10.2019
cd ~/tda_project
python tda_clr_aitchison.py
