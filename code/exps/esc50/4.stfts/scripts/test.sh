#!/bin/bash
#SBATCH --job-name=esc4_8debug.job
#SBATCH --output=esc4_8eraseme.out
#SBATCH --time=04:59:00 
#SBATCH --partition=testing
#SBATCH --mem=15G
#first attempt
 
module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH


python  $HOME/code/exps/esc50/4.stfts/main.py
