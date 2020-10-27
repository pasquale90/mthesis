import pandas as pd
import numpy as np
import torch.optim as optim 
import torch.nn as nn 

#optimizer
def setlr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return optimizer

#decrease learing rate method to reduce overfitting
def change_lr(optimizer, epoch, learning_rate):
  if epoch%10==0:
    new_lr = learning_rate / (10**(epoch//10))
    optimizer = setlr(optimizer, new_lr)
    print(f'Changed learning rate to {new_lr}')
  return optimizer


def reduce_lr(optimizer,learning_rate):
    new_lr = learning_rate / 20
    optimizer = setlr(optimizer, new_lr)
    print(f'Changed learning rate to {new_lr}')
    return optimizer
