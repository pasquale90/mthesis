#!/bin/bash
#SBATCH --job-name=us280debugap.job
#SBATCH --output=us280debugap.out
#SBATCH --time=3:45:00 
#SBATCH --partition=testing
#SBATCH --mem=12G
 
module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH


python  $HOME/code/exps/us8k/2.flat/main.py
