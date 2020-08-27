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
  if (mode == '80'):
    featurepath = paths.featurepath80
  elif (mode == '128'):
    featurepath = paths.featurepath128
  return featurepath

#retrieve extracted features - help function
def get_features(mode, wavname):    
  name= get_featurepath(mode) + wavname.split('.')[0]+'.png'
  image=Image.open(name)
  image=np.array(image)
  image = image/255.0
  return image

class ESC50Data(Dataset):
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
      flatmel=np.ndarray.flatten((get_features(mode, row['filename'])))[np.newaxis,...]    
      self.data.append(flatmel) 
      self.labels.append(self.c2i[row['category']])
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):#load data on demand
    return self.data[idx], self.labels[idx]

