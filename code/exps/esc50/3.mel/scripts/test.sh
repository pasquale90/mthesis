#!/bin/bash
#SBATCH --job-name=3params.job
#SBATCH --output=3params.out
#SBATCH --time=00:59:00 
#SBATCH --partition=testing
#SBATCH --mem=8G
 
module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH


python  $HOME/code/exps/esc50/3.mel/main.py
#python $HOME/testingfolder/sshruntest.py

