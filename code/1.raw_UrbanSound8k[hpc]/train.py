import pandas as pd
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
from optimizer import change_lr
import metrics
import paths
import store
from models import save_model
import attempt
import utils
import console

#train_model_func
def train(model, loss_fn, train_loader, valid_loader,
          epochs, optimizer, learning_rate, device, classes, expid, mode, vfold, modelid):

  exp_attempt = attempt.attempt_class(mode, vfold)
  
  print('Train started..')

  train_losses = []
  valid_losses = []
  
  microrecall = []
  macrorecall = []
  
  microprecision = []
  macroprecision = []
  
  microf1 = []
  macrof1 = []
  
  classf1 = []
  
  
  
  epoch_instance = utils.epochs_class()
  total_epochs = epoch_instance.set_total(epochs)
  epoch = epoch_instance.get_step()

  prevent_overfit = utils.prevent_overfitting()
  
  model_name = expid+'_'+mode+'_v'+str(vfold)+'_a'+str(exp_attempt.get_attempt())+'_'+modelid
  
  console_path = console.makefile(mode,vfold)

  while (epoch<=total_epochs):

    model.train()
    batch_losses=[]

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
    print(f'\nEpoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}\n', file=open(console_path, "a"))
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
    micro_recall,micro_precision,micro, macro_recall,macro_precision,macro = metrics.F1_score(trace_y,trace_yhat,classes)
    
    microrecall.append(micro_recall)
    microprecision.append(micro_precision)
    microf1.append(micro) 

    macrorecall.append(macro_recall)
    macroprecision.append(macro_precision)
    macrof1.append(macro)
    
    #f1 for each class
    f1_class = metrics.F1_Class(trace_y,trace_yhat,classes)
    classf1.append(f1_class)

    #console prints
    print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])}')
    print(f'micro_f1_{epoch} = {micro} ')
    print(f'macro_f1_{epoch} = {macro} ')
    print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])}', file=open(console_path, "a"))
    print(f'micro_f1_{epoch} = {micro} ', file=open(console_path, "a"))
    print(f'macro_f1_{epoch} = {macro} ', file=open(console_path, "a"))
    

    #check if overfitting and reduce lr twice, after that, in case loss is ascending terminate train    
    early_stop = prevent_overfit.detect_overfitting(np.mean(valid_losses[-1]), epoch)
    if early_stop:
      total_epochs=prevent_overfit.early_stopping(epoch_instance)
    
    #always store the best according to avg_micro_and_macro_F1
    current_best, best_micro, best_macro = prevent_overfit.store_best_model(micro,macro)#, best_epoch
    #If achieves current best mean accuracy, Save the model
    if current_best:
      save_model(model_name, model.state_dict())
      #Save analytic results 
      store.save_results(store.define_content(total_epochs,mean_train_losses,mean_valid_losses,microrecall,microprecision,
                              microf1,macrorecall,macroprecision,macrof1,classf1),
                      store.define_filenames_pattern(expid, mode, vfold, exp_attempt.get_attempt()))
      metrics.backup_metrics(trace_y,trace_yhat,classes,paths.results_path+'backup/',mode,vfold)
      
    
    #next epoch
    epoch=epoch_instance.next_step()
  
  #calc_trainable_params
  #params=sum(p.numel() for p in model.parameters() if p.requires_grad)
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
      
  
  
      
  #Save general results so as to quickly compare systems
  genresfilename = store.genres_filename(expid,mode)
  store.save_genres(round(best_micro,3), round(best_macro,3), params, vfold, genresfilename)
  print('params =', params)
  print('params =', params, file=open(console_path, "a"))
  
  exp_attempt.add_attempt()
