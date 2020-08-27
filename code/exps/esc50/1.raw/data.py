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
  return raw


class ESC50Data(Dataset):
  def __init__(self, df, in_col, out_col, audio_path, mode):#(train, 'filename', 'category', mode)
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
      file_path = audio_path+row[in_col]
      raw = get_raw_waveform(file_path,sr=sr)[np.newaxis,...]
      self.data.append(raw) 
      self.labels.append(self.c2i[row['category']])
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):#load data on demand
    return self.data[idx], self.labels[idx]

