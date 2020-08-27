#!/bin/bash
#SBATCH --job-name=us48apfixedinput.job
#SBATCH --output=us48apfixedinput.out
#SBATCH --time=3-23:59:59 
#SBATCH --partition=batch
#SBATCH --mem=25G

module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

python  $HOME/code/exps/us8k/4.stfts/main.py
