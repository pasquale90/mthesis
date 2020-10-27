#!/bin/bash
#SBATCH --job-name=jobid.job
#SBATCH --output=output_filename.out
#SBATCH --time=4-00:00:00 
#SBATCH --partition=batch
#SBATCH --mem=40G

module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

python  $HOME/code/crossexps/us8k/4.stfts/main.py <validation_fold_attribute>
