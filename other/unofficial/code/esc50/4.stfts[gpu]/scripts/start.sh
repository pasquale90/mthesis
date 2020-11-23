#!/bin/bash
#SBATCH --job-name=esc42kfold.job
#SBATCH --output=esc42kfold.out
#SBATCH --time=07:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=5

module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH

python  $HOME/code/crossexps/esc50/4.stfts[gpu]/main.py 2
