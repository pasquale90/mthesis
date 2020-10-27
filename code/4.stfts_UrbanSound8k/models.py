#Import libraries
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F  
#from tqdm import tqdm_notebook as tqdm
import paths
import os
     
class S2Dcnn5(nn.Module):
  def __init__(self, num_cats):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride=1)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 16, kernel_size = 3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(16)
    self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(32)
    self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(32)
    self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.bn5 = nn.BatchNorm2d(64)
    self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.bn6 = nn.BatchNorm2d(64)
    self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn7 = nn.BatchNorm2d(128)
    self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn8 = nn.BatchNorm2d(128)
    self.ap = nn.AdaptiveAvgPool2d((128,32))
    self.dense1 = nn.Linear(128*32,512)
    self.dropout = nn.Dropout(0.1)
    self.dense2 = nn.Linear(512, num_cats)
  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(self.bn1(x))
    x = self.conv2(x)
    x = F.relu(self.bn2(x))
    x = F.max_pool2d(x, kernel_size=2) 
    x = self.conv3(x)
    x = F.relu(self.bn3(x))
    x = self.conv4(x)
    x = F.relu(self.bn4(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = self.conv5(x)
    x = F.relu(self.bn5(x))
    x = self.conv6(x)
    x = F.relu(self.bn6(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = self.conv7(x)
    x = F.relu(self.bn7(x))
    x = self.conv8(x)
    x = F.relu(self.bn8(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = x.view(x.size(0),x.size(1),x.size(2)*x.size(3))
    x = self.ap(x)
    x = x.view(x.size(0),-1)
    x = F.relu(self.dense1(x))
    x = self.dropout(x)
    x = self.dense2(x)
    return x      
                 
#save the model
#PATTERN:exp-num_mode_dataset_'model'_attempt		
def save_model(model_name, vfold, state_dict):
  path = paths.model_savepath
  if not os.path.exists(path):
    os.mkdir(path)
  torch.save(state_dict, path + model_name+'_'+str(vfold)+'.pt')



