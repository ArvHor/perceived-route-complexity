#!/bin/bash
#SBATCH -A HPC2n2025-083
#SBATCH -n 1
#SBATCH -n 28
#SBATCH --time=24:00:00

export DATA_PATH="/proj/nobackup/streetnetwork-alignment/"

module purge > /dev/null 2>1
module load GCC/13.2.0 JupyterLab/4.2.0 SciPy-bundle/2024.05 matplotlib/3.8.2
source $HOME/Public/osmnx201_venv/bin/activate
jupyter lab --no-browser --ip $(hostname)