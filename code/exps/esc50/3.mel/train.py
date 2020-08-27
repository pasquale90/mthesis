import pandas as pd
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torch.utils.data import Dataset, DataLoader 
#from tqdm import tqdm_notebook as tqdm

#import data
#device=main.device
import metrics
import paths
import store
from model import save_model
import attempt
import utils

#train_model_func
def train(model, loss_fn, train_loader, valid_loader,
          epochs, optimizer, change_lr , learning_rate, device, classes, expid, mode, vfold, modelid):

  exp_attempt = attempt.attempt_class(mode, vfold)
  
  print('Train started..')

  train_losses = []
  valid_losses = []
  #accuracies = []
  #micro_aurocs = []
  #macro_aurocs = []
  microf1 = []
  macrof1 = []
  classf1 = []
  
  epoch_instance = utils.epochs_class()
  total_epochs = epoch_instance.set_total(epochs)
  epoch = epoch_instance.get_step()


  overfit = utils.prevent_overfitting()
 
 # for epoch in tqdm(range(1,epochs+1)):
 # for epoch in range(1,epochs+1):
  #Train step
  while (epoch<=total_epochs):

    model.train()
    batch_losses=[]
    #if change_lr:
    #  optimizer = change_lr(optimizer, epoch, learning_rate)
    #for i, data in tqdm(enumerate(train_loader)): 
    for i,data in enumerate(train_loader):
      x, y = data
      optimizer.zero_grad()
      x = x.to(device, dtype=torch.float32)
      y = y.to(device, dtype=torch.long) 
      y_hat = model(x) 
      loss = loss_fn(y_hat, y)
      loss.backward()
      batch_losses.append(loss.item())						
      optimizer.step()
    train_losses.append(batch_losses)
    mean_train_losses=([np.mean(l) for l in train_losses])
    
    print()
    print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
    print()
    
    #Validation step
    model.eval()
    
    batch_losses=[]
    trace_y = []
    trace_yhat = []
    
    for i, data in enumerate(valid_loader):
      x, y = data
      x = x.to(device, dtype=torch.float32)
      y = y.to(device, dtype=torch.long)
      y_hat = model(x)
      loss = loss_fn(y_hat, y)
      trace_y.append(y.cpu().detach().numpy())
      trace_yhat.append(y_hat.cpu().detach().numpy())      
      batch_losses.append(loss.item())
    valid_losses.append(batch_losses)
    mean_valid_losses=([np.mean(l) for l in valid_losses])
    
    trace_y = np.concatenate(trace_y)
    trace_yhat = np.concatenate(trace_yhat)

    #f1,micro,macro
    micro, macro = metrics.F1_score(trace_y,trace_yhat)
    microf1.append(micro) 
    macrof1.append(macro)
    
    #f1 for each class
    f1_class = metrics.F1_Class(trace_y,trace_yhat,classes)
    classf1.append(f1_class)

    #console prints
    print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])}')
    print(f'micro_f1_{epoch} = {micro} ')
    print(f'macro_f1_{epoch} = {macro} ')
    #print(f'micro_auroc_{epoch} = {micro_auroc} ')
    #print(f'macro_auroc_{epoch} = {macro_auroc} ')
  
    #check if overfitting and reduce lr twice, after that, in case loss is ascending terminate train    
    early_stop, optimizer = overfit.detect_overfitting(np.mean(valid_losses[-1]),optimizer, learning_rate, epoch_instance)
    if early_stop:
      total_epochs=epoch_instance.set_total(epoch_instance.get_step())
    epoch=epoch_instance.next_step()

  #Save results 
  store.save_results(store.define_content(total_epochs,
		                                     mean_train_losses,
                                         mean_valid_losses, 
                                         microf1, 
                                         macrof1,
		                                     classf1),
                      store.define_filenames_pattern(expid, mode, vfold, exp_attempt.get_attempt()))
    
  #calc_trainable_params
  params=sum(p.numel() for p in model.parameters() if p.requires_grad)

  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	
  params_2 = sum([np.prod(p.size()) for p in model_parameters])

  #Save general results so as to quickly compare systems
  genres_filename = store.genres_filename(expid,mode)
  store.save_genres(micro, macro, params, vfold, genres_filename)
  print('params =', params)
  print('params2 = ',params_2)

  #Save the model
  model_name = expid+'_'+mode+'_v'+str(vfold)+'_a'+str(exp_attempt.get_attempt())+'_'+modelid
  save_model(model_name,model.state_dict())

  exp_attempt.add_attempt()   
