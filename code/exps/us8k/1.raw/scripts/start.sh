#!/bin/bash
#SBATCH --job-name=us144ap.job
#SBATCH --output=us144ap.out
#SBATCH --time=5-23:59:59 
#SBATCH --partition=batch
#SBATCH --mem=20G

module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

python  $HOME/code/exps/us8k/1.raw/main.py
