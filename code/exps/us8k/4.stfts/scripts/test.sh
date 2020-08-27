#!/bin/bash
#SBATCH --job-name=us48test.job
#SBATCH --output=us48test.out
#SBATCH --time=00:09:00 
#SBATCH --partition=testing
#SBATCH --mem=12G
 
module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH


python  $HOME/code/exps/us8k/4.stfts/main.py
