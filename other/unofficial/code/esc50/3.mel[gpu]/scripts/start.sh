#!/bin/bash
#SBATCH --job-name=esc3_360kfold.job
#SBATCH --output=esc3_360kfold.out
#SBATCH --time=07:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=35G
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=10

module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

python  $HOME/code/crossexps/esc50/3.mel[gpu]/main.py 360