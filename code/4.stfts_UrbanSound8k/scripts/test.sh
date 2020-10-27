#!/bin/bash
#SBATCH --job-name=test_id.job
#SBATCH --output=output_filename.out
#SBATCH --time=04:00:00 
#SBATCH --partition=testing
#SBATCH --mem=7G
 
module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

python $HOME/code/crossexps/us8k/4.stfts/main.py 3
