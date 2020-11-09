#!/bin/bash
#SBATCH --job-name=usdebug.job
#SBATCH --output=debug.out
#SBATCH --time=3:29:00 
#SBATCH --partition=testing
#SBATCH --mem=5G
 
module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH


python  $HOME/code/crossexps/us8k/1.raw[gpu]/main.py 16
