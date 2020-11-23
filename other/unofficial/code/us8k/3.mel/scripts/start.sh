#!/bin/bash
#SBATCH --job-name=us3_360kfold.job
#SBATCH --output=us3_360kfold.out
#SBATCH --time=07:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=10

module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

python  $HOME/code/crossexps/us8k/3.mel/main.py 360
