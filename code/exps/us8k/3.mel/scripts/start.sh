#!/bin/bash
#SBATCH --job-name=us3128finbatch4.job
#SBATCH --output=us3128fixedinput.out
#SBATCH --time=2-23:59:59 
#SBATCH --partition=batch
#SBATCH --mem=20G

module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

python  $HOME/code/exps/us8k/3.mel/main.py
