#!/bin/bash
#SBATCH --job-name=us3debug.job
#SBATCH --output=debug.out
#SBATCH --time=03:10:00 
#SBATCH --partition=testing
#SBATCH --mem=12G
 
module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH


#python  $HOME/code/crossexps/us8k/3.mel/_test.py.py
python -u $HOME/code/crossexps/us8k/3.mel/main.py > check $HOME/flush.out
