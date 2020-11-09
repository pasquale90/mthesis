#!/bin/bash
#SBATCH --job-name=con.job
#SBATCH --output=debug.out
#SBATCH --time=00:00:30 
#SBATCH --partition=testing
#SBATCH --mem=5K
 
module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

#rm -r $HOME/code/crossexps/us8k/1.raw[gpu]/console
python  $HOME/code/crossexps/us8k/1.raw[gpu]/test.py
