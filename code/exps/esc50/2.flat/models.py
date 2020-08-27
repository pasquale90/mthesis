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

#ESC50_MODEL_80_flatmelcnn
class  esc_flat_model_hybrid(nn.Module):
    def __init__(self, mode, input_shape, batch_size=16, num_cats=50):
        super(esc_flat_model_hybrid, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
  
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3),
            nn.Dropout(0.5))

        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))

        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))

        self.conv10 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU())

        self.conv11 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc = nn.Linear(512, num_cats)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = x.view(x.shape[0], 1,-1 )


        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)

        x = x.view(x.shape[0], x.size(1) * x.size(2))
        x = self.fc(x)
        return x

#ESC50_MODEL_128_flatmelcnn
class  esc_flat_model_128(nn.Module):
    def __init__(self,input_shape, batch_size=16, num_cats=50):
        super(esc_flat_model_128, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
  
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3),
            nn.Dropout(0.5))

        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))

        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))

        self.conv10 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU())

        self.conv11 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc = nn.Linear(512, num_cats)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = x.view(x.shape[0], 1,-1 )


        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = x.view(x.shape[0], x.size(1) * x.size(2))
        x = self.fc(x)
        return x

#ad pool model_cnn
class hybridap1dcnn(nn.Module):
    def __init__(self,batch_size=1, num_cats=50):
        super(hybridap1dcnn, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
  
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))


        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        
        self.conv6 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU())

        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.ap = nn.AdaptiveMaxPool1d(5)
        #self.ap = nn.AdaptiveAvgPool1d(5)

        self.fc = nn.Linear(512*5, num_cats)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        #print('input',x.shape)
        x = x.view(x.shape[0], 1,-1 )
        #print('view',x.shape)
        x = self.conv1(x)
        #print('1',x.shape)
        x = self.conv2(x)
        #print('2',x.shape)
        x = self.conv3(x)
        #print('3',x.shape)
        x = self.conv4(x)
        #print('4',x.shape)
        x = self.conv5(x)
        #print('5',x.shape)
        x = self.conv6(x)
        #print('6',x.shape)
        x = self.conv7(x)
        #print('7',x.shape)

        x = self.ap(x)
        #print('ap',x.shape)

        x = x.view(x.shape[0], x.size(1) * x.size(2))
        #print('view',x.shape)
        x = self.fc(x)
        #print('densce output',x.shape)
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
