#!/bin/bash
#SBATCH --job-name=esc3128apall.job
#SBATCH --output=esc3128apall.out
#SBATCH --time=1-23:59:59 
#SBATCH --partition=batch
#SBATCH --mem=12G

module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

python  $HOME/code/exps/esc50/3.mel/main.py
