#!/bin/bash
#SBATCH --job-name=esc2_debugap.job
#SBATCH --output=esc2debugap.out
#SBATCH --time=00:05:00 
#SBATCH --partition=testing
#SBATCH --mem=8G
 
module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH


python  $HOME/code/exps/esc50/2.flat/main.py


