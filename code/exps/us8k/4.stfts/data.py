#Import libraries
import os
import pandas as pd
import librosa
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
#from tqdm import tqdm_notebook as tqdm

import paths

def get_featurepath(mode):
  if (mode == 'exp4'):
    featurepath = paths.featurepath4
  elif (mode == 'exp8'):
    featurepath = paths.featurepath8
  return featurepath

#retrieve extracted features - help function
def get_features(mode, wavname):    
  name= get_featurepath(mode) + wavname.split('.')[0]+'.png'
  image=Image.open(name)
  image=np.array(image)
  image = image/255.0
  
  #dublicate small_size features
  if image.shape[1]<(image.shape[0]//2):
    copy = image
    while(image.shape[1]<(image.shape[0])):
      image = np.concatenate((image,copy),axis=1)
      
  return image

class US8KData(Dataset):
  def __init__(self, df, in_col, out_col, mode):#(train, 'filename', 'category', mode)
    self.df = df
    self.data = []
    self.labels = []
    self.c2i={}
    self.i2c={}
    self.categories = sorted(df[out_col].unique())
    for i, category in enumerate(self.categories):
      self.c2i[category]=i
      self.i2c[i]=category
    #for ind in tqdm(range(len(df))):
    for ind in range(len(df)):
      row = df.iloc[ind]
      self.data.append(get_features(mode, row['slice_file_name'])[np.newaxis,...])
      self.labels.append(self.c2i[row['Class']])
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):#load data on demand
    return self.data[idx], self.labels[idx]
