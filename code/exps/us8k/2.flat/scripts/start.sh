#!/bin/bash
#SBATCH --job-name=us2128apfixedinput.job
#SBATCH --output=us2128apfixedinput.out
#SBATCH --time=3-23:59:59 
#SBATCH --partition=batch
#SBATCH --mem=20G

module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

python  $HOME/code/exps/us8k/2.flat/main.py
