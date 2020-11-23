import os

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")

#data.py
data_path=HOME+"/datasets/esc-50/esc50.csv"
audio_path=HOME+"/datasets/esc-50/audio/"

#model.py		
model_savepath=EXP+"/saved_models/"	

#store.py
results_path=EXP+"/results/"
compare_results_path = EXP+"/results/gathered/"	

#attempt.py
attempt_path=EXP+"/expattempt/"

