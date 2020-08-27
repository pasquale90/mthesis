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

#from tqdm import tqdm_notebook as tqdm
#from model import esc_mel_model_hybrid
import models
from optimizer import setlr, lr_decay
import train
import data
from mode import mode_class
import paths


#define experiment
expid='esc2'

#define mode
mode_instance = mode_class(128)
mode=mode_instance.get_mode()
print(mode)

#define paths
data_path=paths.data_path
audio_path=paths.audio_path

#Import Dataset                      
esc50 = pd.read_csv(data_path)
audiofiles = os.listdir(audio_path)
print(esc50.shape)
print(f'audio_files length:{len(audiofiles)}')

#store_sorted_class_names, in the same way that are returned from dataset_class in data.py
esc_classes = sorted(esc50['category'].unique())

#train and fold definition
vfold=5 #4,3,2,1
train_data = esc50[esc50['fold']!=vfold]
valid_data = esc50[esc50['fold']==vfold]

#load spectograms
print('loadin mel-spectograms..')
train_data = data.ESC50Data(train_data, 'filename', 'category', mode)
valid_data = data.ESC50Data(valid_data, 'filename', 'category', mode)
print('features are loaded')

#data iterator
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=True)

#define device
if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')
print(device)
                                                                                                          
'''
if(mode=='80'):
  model = models.esc_flat_model_hybrid(mode, input_shape=(1,(80*431)), batch_size=16, num_cats=50).to(device)
elif(mode=='128'):
  model = models.esc_flat_model_hybrid(mode, input_shape=(1,(128*431)), batch_size=16, num_cats=50).to(device)
'''
#init model
'''
model = models.esc_flat_model_hybrid(mode, input_shape=(1,int(mode),431), batch_size=16, num_cats=50).to(device)
'''
model = models.hybridap1dcnn(batch_size=16,num_cats=50).to(device)
modelid='hybridap1dcnn'
print(f'model_{modelid} initialized...')

#define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
learning_rate = 2e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 100

#train
train.train(model, loss_fn, train_loader, valid_loader, 
                            epochs, optimizer, lr_decay, learning_rate, device, esc_classes, expid, mode, vfold, modelid)

                                                                                                    

