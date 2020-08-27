#!/bin/bash
#SBATCH --job-name=us38test.job
#SBATCH --output=us38test.out
#SBATCH --time=5:59:00 
#SBATCH --partition=testing
#SBATCH --mem=12G
 
module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH


python  $HOME/code/exps/us8k/3.mel/main.py
