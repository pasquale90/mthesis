#!/bin/bash
#SBATCH --job-name=us1.job
#SBATCH --output=us1.out
#SBATCH --time=12:59:00 
#SBATCH --partition=testing
#SBATCH --mem=15G
 
module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH


python  $HOME/code/exps/us8k/1.raw/main.py
