#!/bin/bash
#SBATCH --job-name=esc2128ap.job
#SBATCH --output=esc2128ap.out
#SBATCH --time=4-23:59:59 
#SBATCH --partition=batch
#SBATCH --mem=15G

module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

python  $HOME/code/exps/esc50/2.flat/main.py
