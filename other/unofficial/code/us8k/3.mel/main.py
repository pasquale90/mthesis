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
import folds
import features
import augmentation
import kfold

#define expid
expid='us3'

#define mode
mode_instance = mode_class(int(sys.argv[1]))
mode=mode_instance.get_mode()
print(mode)

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


print(us8k.columns)
us8k.rename(columns={'class':'Class'}, inplace=True)
print('\n\ncolumn <class> became... <%s>'%us8k.columns[-1])

#store_sorted_classes in the same way as defined in data.py
us8k_classes = sorted(us8k['Class'].unique())
num_classes = len(us8k_classes)
print('num_classes: ',num_classes)

'''
#train and fold definition
vfold = 10
train_folds = [folds.index(fold)+1 for i,fold in enumerate(folds) if vfold!=folds.index(fold)+1]
#train_folds = [folds.index(fold)+1 for i,fold in enumerate(folds) if (vfold-1)==folds.index(fold)+1]#for testing
valid_folds = [folds.index(fold)+1 for i,fold in enumerate(folds) if vfold==folds.index(fold)+1]
print('train_folds: ',train_folds)
print('valid_folds: ',valid_folds)
'''

sampling_rate, hop_length, fft_points, mel_bands = features.analysis_parameters(mode)
print(f'sampling_rate: {sampling_rate}, hop_length: {hop_length}, fft_points: {fft_points}, mel_bands: {mel_bands}')

#extract_features
features, labels, folders = features.extract_mel_spectogram(audio_path,us8k,folds,audiofiles,sampling_rate,hop_length,fft_points,mel_bands)
print(f' feature\'s len : {len(features)}, labels : {len(labels)}, folders : {len(folders)}')


#define device
if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')
print('device : ',device)

'''
#straight forward k-fold validation
for i in range(10):
 vfold=i+1
 if (vfold==9 or vfold==10):
  train_folds,valid_fold = kfold.k_fold(vfold,folds)
  train_loader, valid_loader = kfold.reload_data(features,labels,folders,train_folds,valid_fold, batch_size=16)
  model, loss_fn, epochs, learning_rate, optimizer, modelid = kfold.reinitialize_model(num_classes, device)
  train.train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, learning_rate, device, us8k_classes, expid, mode, vfold, modelid)
  
  print('\n')
  print('.......................................................................................................')
  print('\n')
  #print('Changing validation fold', file=open(paths.console_path, "a"))
  print('\n')
  print('.......................................................................................................')
  print('\n')
'''                    
'''                                                                      
if __name__ == '__main__':
    num_processes = 2
    batch_size = 16
    for rank in range(num_processes):
        #vfold = rank+1
        vfold=rank+9
        p = mp.Process(target=kfold.cross_validation, args=(vfold,folds,features,labels,folders,batch_size,num_classes, device,us8k_classes,expid, mode))
        p.start()
'''
for i in range(10):
 vfold=i+1
 if (vfold>=7): 
  batch_size = 16 
  kfold.cross_validation(vfold,folds,features,labels,folders,batch_size,num_classes, device,us8k_classes,expid, mode)
  
