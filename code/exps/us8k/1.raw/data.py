#Import libraries
import os
import pandas as pd
import librosa
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
#from tqdm import tqdm_notebook as tqdm

import paths

def get_raw_waveform(file_path,sr):
  raw, fs=librosa.load(file_path,sr=sr, mono=True)
  length = raw.shape[0]
  if length<(sr//2):
    raw=zero_pad(raw,sr)
  return raw

def zero_pad(signal,sr):
  num_zeros=(sr//2)-len(signal)
  zp=np.zeros(num_zeros,dtype=float)
  padded_signal = np.concatenate((signal,zp),axis=0)
  return padded_signal

class US8KData(Dataset):
  def __init__(self, df, in_col, out_col, audio_path, folds, mode):
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
      sr = int(mode)
      file_path = audio_path+folds[row['fold']-1]+'/'+row[in_col]
      raw = get_raw_waveform(file_path,sr=sr)[np.newaxis,...]
 
      self.data.append(raw)  
      self.labels.append(self.c2i[row['Class']])
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):#load data on demand
    return self.data[idx], self.labels[idx]


