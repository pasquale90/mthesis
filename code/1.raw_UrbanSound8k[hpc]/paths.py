import os

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")

#data.py
data_path=HOME+"/datasets/urbansound8k/UrbanSound8K/metadata/UrbanSound8K.csv"
audio_path=HOME+"/datasets/urbansound8k/UrbanSound8K/audio/"

#model.py		
model_savepath=EXP+"/saved_models/"	

#store.py
results_path=EXP+"/results/"
compare_results_path = EXP+"/results/gathered/"	

#attempt.py
attempt_path=EXP+"/expattempt/"

#console
console_path = EXP + "/console/"
