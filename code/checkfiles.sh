#!/bin/bash
#SBATCH --job-name=checkfiles.job
#SBATCH --output=checkfiles.out
#SBATCH --time=00:30:00
#SBATCH --partition=testing
#SBATCH --mem=8G

module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH


python  $HOME/code/checkfiles.py
