#Import libraries
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
#import torch.optim as optim 
#from torch.utils.data import Dataset, DataLoader 
#from tqdm import tqdm_notebook as tqdm
#from params import device
import paths
import os


class ESC50_MODEL_2_rawcnn(nn.Module):
    def __init__(self):#input_shape=5x16000=80000, batch_size=16, num_cats=50
        super(ESC50_MODEL_2_rawcnn, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8, padding=1) 
        self.conv2 = nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=8, padding=1) 
        self.conv3 = nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=1) 
        self.bn4 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 512)
        self.drop2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 50)

        
    def forward(self, x):
        x = x.view(x.shape[0], 1,-1 )
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = x.view(x.shape[0], 1, x.size(1) * x.size(2))
        x = self.fc1(self.drop1(x))
        x = self.fc2(self.drop2(x))
        x = F.log_softmax(x, dim = 2)
        x = x.view(x.shape[0], x.size(1)*x.size(2))
        return x


#save the model
#PATTERN:exp-num_mode_dataset_'model'_attempt		
def save_model(model_name, state_dict):
  path = paths.model_savepath
  if not os.path.exists(path):
    os.mkdir(path)
  torch.save(state_dict, path + model_name+'pth')

#load model
def load_model(model_name, mode):
  if (mode=='80'):
    model = esc_mel_model_hybrid(input_shape=(1,80,431), batch_size=16, num_cats=50)
  elif(mode=='128'):
    model = esc_mel_model_hybrid(input_shape=(1,128,431), batch_size=16, num_cats=50)
  model.load_state_dict(torch.load(paths.model_savepath+model_name+'.pth'))

