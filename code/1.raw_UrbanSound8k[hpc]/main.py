#runs the scripts

#Import libraries
import os
import pandas as pd
import librosa
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch.multiprocessing as mp

import sys

import models
from optimizer import setlr, change_lr
import train
import data
from mode import mode_class
import paths
import folds
import preprocess
import augmentation
import kfold

#define expid
expid='us1'

#define mode
expmode = int(sys.argv[1])
mode_instance = mode_class(expmode)
mode=mode_instance.get_mode()
print('mode : ', mode)

#define paths
data_path=paths.data_path
audio_path=paths.audio_path

#Import Dataset                      
us8k = pd.read_csv(data_path)
print('csv containts {} rows and {} columns.'.format(len(us8k.index),len(us8k.columns)))

folds, audiofiles, order, filesum = folds.read_folds(audio_path, folds)

for f in folds:
    print('fold no_%d contains %d audiofiles'%(order[f],len(audiofiles[f]))) 
print('All in all there are %d audio files found in 8k Urban Sound dataset folders'%filesum)

#rename class to Class
print(us8k.columns)
us8k.rename(columns={'class':'Class'}, inplace=True)
print('\n\ncolumn <class> became... <%s>'%us8k.columns[-1])

#store_sorted_class_names, in the same way that are returned from dataset_class in data.py
us8k_classes = sorted(us8k['Class'].unique())#defined as in data.py
print(f'classes{us8k_classes}')
num_classes = len(us8k_classes)
print('num_classes: ',num_classes)

#get analysis parameters
sampling_rate, window_size, hop_length = preprocess.analysis_parameters(mode)
print(f'sampling_rate: {sampling_rate}, window_size: {window_size}, hop_length: {hop_length}')

#extract_features       
features, labels, folders = preprocess.preprocess_data(audio_path, us8k, folds, audiofiles, mode)
print(f' feature\'s len : {len(features)}, labels : {len(labels)}, folders : {len(folders)}')

#define 
if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')
print(device)

if __name__ == '__main__':
    num_processes = 10
    batch_size = 16
    for rank in range(num_processes):
	vfold = rank+1
        p = mp.Process(target=kfold.cross_validation, args=(vfold,folds,features,labels,folders,batch_size,num_classes, device,us8k_classes,expid, mode))
        p.start()
