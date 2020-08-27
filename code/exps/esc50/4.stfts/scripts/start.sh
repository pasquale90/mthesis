#!/bin/bash
#SBATCH --job-name=esc48ap.job
#SBATCH --output=esc48ap.out
#SBATCH --time=4-23:59:59 
#SBATCH --partition=batch
#SBATCH --mem=15G

module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

python  $HOME/code/exps/esc50/4.stfts/main.py
