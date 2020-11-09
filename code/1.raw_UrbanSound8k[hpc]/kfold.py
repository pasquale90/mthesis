import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader 

import data
import models

#train and evaluation folds definition
def k_fold(vfold,folds):
  train_folds = [folds.index(fold)+1 for i,fold in enumerate(folds) if vfold!=folds.index(fold)+1]
  valid_fold = [folds.index(fold)+1 for i,fold in enumerate(folds) if vfold==folds.index(fold)+1]
  print('train_folds: ',train_folds)
  print('valid_fold: ',valid_fold)
  return train_folds,valid_fold

def reload_data(features,labels,folders,train_folds,valid_fold, batch_size=16):

  #load splits    
  train_data = data.Data(features,labels,folders,train_folds)
  valid_data = data.Data(features,labels,folders,valid_fold)
  print('features are loaded') 

  #batch_size = 16	
  print(f'batch_size: {batch_size}')

  #data iterator
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False) 
  return train_loader, valid_loader


def reinitialize_model(num_classes,device):

  #introduce reproducibility
  seed = 70
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  #init model
  model = models.R1Dcnn9(num_cats=num_classes).to(device)
  modelid= 'R1Dcnn9'
  total_params = sum(p.numel() for p in model.parameters())
  print(f'model_{modelid} initialized with total : {total_params} parameters.')

  loss_fn = nn.CrossEntropyLoss()
  epochs = 25
  learning_rate = 1e-5
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  print(f' learning_rate = {learning_rate}, epochs = {epochs}')
  print(f' epochs = {epochs}')
  
  return model, loss_fn, epochs, learning_rate, optimizer, modelid


#straight forward k-fold validation
def cross_validation(vfold,folds,features,labels,folders,batch_size,num_classes, device,us8k_classes,expid, mode):

  train_folds,valid_fold = kfold.k_fold(vfold,folds)
  train_loader, valid_loader = kfold.reload_data(features,labels,folders,train_folds,valid_fold, batch_size=16)
  model, loss_fn, epochs, learning_rate, optimizer, modelid = kfold.reinitialize_model(num_classes, device)
  train.train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, learning_rate, device, us8k_classes, expid, mode, vfold, modelid)
  
  print('\n')
  print('.......................................................................................................')
  print('\n')
  print('Changing validation fold', file=open(console_path, "a"))
  print('\n')
  print('.......................................................................................................')
  print('\n')
