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
from tqdm import tqdm_notebook as tqdm

import model
import models#adaptivepoolingtry
from optimizer import setlr, lr_decay
import train
import data
from mode import mode_class
import paths
import folds


#define expid
expid='us2'

#define mode
mode_instance = mode_class(80)
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
print(us8k_classes)


#train and fold definition
vfold=10
train_data = us8k[us8k['fold']!=vfold]
valid_data = us8k[us8k['fold']==vfold]

#load spectograms
print('loadin mel-spectograms..')
train_data = data.US8KData(train_data, 'slice_file_name', 'Class', mode)
valid_data = data.US8KData(valid_data, 'slice_file_name', 'Class', mode)
print('features are loaded')


#data iterator
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False)

#define device
if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')
print(device)

'''                                                                                                          
#init model
if(mode=='80'):
   model = model.us8k_mel_model_hybrid(mode, input_shape=(1,80,431), batch_size=16, num_cats=50).to(device)
elif(mode=='128'):
   model = model.us8k_mel_model_hybrid(mode, input_shape=(1,128,431), batch_size=16, num_cats=50).to(device)
 #  model = model.us8k_mel_model_hybrid(mode, input_shape=(1,128,345),batch_size=16, num_cats=50).to(device)
'''
model = models.hybridap1dcnn(batch_size=1,num_cats=10).to(device)
modelid='hybridap1dcnn'

print('model initialized...')

#define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
learning_rate = 2e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 100

#train
train.train(model, loss_fn, train_loader, valid_loader, 
                            epochs, optimizer, lr_decay, learning_rate, device, us8k_classes, expid, mode, vfold, modelid)

                                                                                                    

