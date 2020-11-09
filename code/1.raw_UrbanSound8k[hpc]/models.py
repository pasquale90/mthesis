#Import libraries
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import paths
import os

class R1Dcnn9(nn.Module):
    def __init__(self,num_cats):
        super(R1Dcnn9, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=64, stride=2)#64
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=64, stride=2)#64
        self.bn2 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=4) 
        self.conv3 = nn.Conv1d(32, 64, kernel_size=32, stride=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=32, stride=2)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=2) 
        self.conv5 = nn.Conv1d(64, 128, kernel_size=16, stride=2) 
        self.bn5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=8, stride=2) 
        self.bn6 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=2) 
        self.conv7 = nn.Conv1d(128, 256, kernel_size=4, stride=2) 
        self.bn7 = nn.BatchNorm1d(256)
        self.conv8 = nn.Conv1d(256, 256, kernel_size=4, stride=2) 
        self.bn8 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2) 
        self.drop1 = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool1d(7)
        self.fc1 = nn.Linear(256*7, 512)
        self.drop2 = nn.Dropout(0.16)
        self.fc2 = nn.Linear(512, num_cats)
        
    def forward(self, x):
        #print(x.shape)

        x = x.view(x.shape[0], 1,-1 )
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool2(x)
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        x = self.pool3(x)
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.conv8(x)
        x = F.relu(self.bn8(x))
        x = self.pool4(x)
        x = self.ap(x)
        x = x.view(x.shape[0], x.size(1) * x.size(2))
        x = self.relu(self.fc1(self.drop1(x)))
        x = self.fc2(self.drop2(x))
        x = F.log_softmax(x, dim = 1)
        return x    

                                                                                        
#save the model
#PATTERN:exp-num_mode_dataset_'model'_attempt		
def save_model(model_name, state_dict):
  path = paths.model_savepath
  if not os.path.exists(path):
    os.mkdir(path)
  torch.save(state_dict, path + model_name+'.pt')

#load model
def load_model(model_name, mode):
  if (mode=='80'):
    model = esc_mel_model_hybrid(input_shape=(1,80,431), batch_size=16, num_cats=50)
  elif(mode=='128'):
    model = esc_mel_model_hybrid(input_shape=(1,128,431), batch_size=16, num_cats=50)
  model.load_state_dict(torch.load(paths.model_savepath+model_name+'.pth'))

