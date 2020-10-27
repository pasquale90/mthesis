#runs the scripts

#Import libraries
import os
import pandas as pd
import librosa
#from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
#import librosa.display
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader 
import sys

#from tqdm import tqdm_notebook as tqdm
#from model import esc_mel_model_hybrid
import models
from optimizer import setlr, change_lr
import train
import data
from mode import mode_class
import paths
import folds
import features
import augmentation

#define expid
expid='us4'

#define mode
#exp4|exp8
mode_instance = mode_class(2)
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
sampling_rate, hop_length, fft_points = features.analysis_parameters(mode)
print(f'sampling_rate: {sampling_rate}, hop_length: {hop_length}, fft_points: {fft_points}')

#extract_features
features, labels, folders = features.extract_stft_spectogram(audio_path,us8k,folds,audiofiles,sampling_rate,hop_length,fft_points)
print(f' feature\'s len : {len(features)}, labels : {len(labels)}, folders : {len(folders)}')

#train and fold definition
vfold = int(sys.argv[1])
train_folds = [folds.index(fold)+1 for i,fold in enumerate(folds) if vfold!=folds.index(fold)+1]
#train_folds = [folds.index(fold)+1 for i,fold in enumerate(folds) if (vfold-1)==folds.index(fold)+1]#for testing
valid_folds = [folds.index(fold)+1 for i,fold in enumerate(folds) if vfold==folds.index(fold)+1]
print('train_folds: ',train_folds)
print('valid_folds: ',valid_folds)

#load spectograms
train_data = data.Data(features,labels,folders,train_folds,augmentation.train_transforms)
valid_data = data.Data(features,labels,folders,valid_folds,augmentation.valid_transforms)
print('features are loaded')

batch_size = 16
print(f'batch_size: {batch_size}')

#data iterator
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

#define device
if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')
print(device)
                                                                                                          
#introduce reproducibility
seed = 7
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#init model
#model = models.S2Dcnn5(num_cats=num_classes).cuda()#to(device)
model =  models.S2Dcnn5(num_cats=num_classes).to(device)
modelid='S2Dcnn5'
total_params = sum(p.numel() for p in model.parameters())
print(f'model_{modelid} initialized with total : {total_params} parameters.')

#define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-5
epochs = 34
print(f' learning_rate = {learning_rate}, epochs = {epochs}')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#train - fold1
train.train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, learning_rate, device, us8k_classes, expid, mode, vfold, modelid)
