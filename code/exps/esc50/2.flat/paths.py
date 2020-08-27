import os

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")

#data.py
data_path=HOME+"/datasets/esc-50/esc50.csv"
audio_path=HOME+"/datasets/esc-50/audio/"
featurepath80=HOME+"/features/esc-50/mel/80/"
featurepath128=HOME+"/features/esc-50/mel/128/"

#model.py		
model_savepath=EXP+"/saved_models/"	

#store.py
results_path=EXP+"/results/"
compare_results_path = HOME+"/results/"	

#attempt.py
attempt_path=EXP+"/expattempt/"

