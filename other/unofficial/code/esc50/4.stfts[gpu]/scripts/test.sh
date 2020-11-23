#!/bin/bash
#SBATCH --job-name=erase_results.job
#SBATCH --output=erase_res.out
#SBATCH --time=00:00:10 
#SBATCH --partition=testing
#SBATCH --mem=10K
 
module load gcc miniconda3 
source $CONDA_PROFILE/conda.sh
conda activate $HOME/melEnv
export PATH=$CONDA_PREFIX/bin:$PATH


#python  $HOME/code/crossexps/us8k/3.mel/_test.py.py
#python -u $HOME/code/crossexps/us8k/3.mel/main.py > check $HOME/flush.out
rm -r $HOME/code/crossexps/esc50/4.stfts[gpu]/results
