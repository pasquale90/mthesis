import os

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")

#data.py
data_path=HOME+"/datasets/urbansound8k/UrbanSound8K/metadata/UrbanSound8K.csv"
audio_path=HOME+"/datasets/urbansound8k/UrbanSound8K//audio/"
featurepath4=HOME+"/features/ur8k/stfts/exp4/"
featurepath8=HOME+"/features/ur8k/stfts/exp8/"

#model.py		
model_savepath=EXP+"/saved_models/"	

#store.py
results_path=EXP+"/results/"	
compare_results_path = HOME+"/results/"
#attempt.py
attempt_path=EXP+"/expattempt/"

