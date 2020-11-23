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
from torch.utils.data import DataLoader 
import sys 
import torch.multiprocessing as mp

import models
from optimizer import setlr, change_lr
import train
import data
from mode import mode_class
import paths
import features
import augmentation
import kfold

#define expid
expid='esc3'

#define mode
mode_instance = mode_class(int(sys.argv[1]))
mode=mode_instance.get_mode()
print(mode)

#define paths
data_path=paths.data_path
audio_path=paths.audio_path

#Import Dataset                      
esc50 = pd.read_csv(data_path)
audiofiles = os.listdir(audio_path)
print('dataset\'s shape : ',esc50.shape)
print(f'audio_files length : {len(audiofiles)}')

#store_sorted_class_names, in the same way that are returned from dataset_class in data.py
esc_classes = sorted(esc50['category'].unique())#defined as in data.py
num_classes = len(esc_classes)
print('num_classes: ',num_classes)

folds = sorted(esc50['fold'].unique())
print('folds : ', folds)

sampling_rate, hop_length, fft_points, mel_bands = features.analysis_parameters(mode)
print(f'sampling_rate: {sampling_rate}, hop_length: {hop_length}, fft_points: {fft_points}, mel_bands: {mel_bands}')

#extract_features
features, labels, folders = features.extract_mel_spectogram(audio_path,esc50,audiofiles,sampling_rate,hop_length,fft_points,mel_bands)
print(f' feature\'s len : {len(features)}, labels : {len(labels)}, folders : {len(folders)}')


#define device
if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')
print('device : ',device)

      
'''                                                                      
if __name__ == '__main__':
    num_processes = 2
    batch_size = 16
    for rank in range(num_processes):
        #vfold = rank+1
        vfold=rank+9
        p = mp.Process(target=kfold.cross_validation, args=(vfold,folds,features,labels,folders,batch_size,num_classes, device,esc_classes,expid, mode))
        p.start()
'''
for i in range(5):
  vfold=i+1
  #if (vfold>=7): 
  batch_size = 16 
  kfold.cross_validation(vfold,folds,features,labels,folders,batch_size,num_classes,device,esc_classes,expid, mode)
  
