#Import libraries
import numpy as np
import torch 
from torch.utils.data import Dataset

class Data(Dataset): 
  def __init__(self, features,labels,folders,split):

    def convert_to_tensor(data):
      tensors = []
      for file in data:
        tensor = torch.Tensor(list(file))
        tensors.append(tensor)
      return tensors
      
    print('Loading_features......')

    #features, labels, folders
    self.indexes = [i for i, val in enumerate(folders) if val in split]
    self.data = [features[x] for x in self.indexes]
    self.labels = [labels[x] for x in self.indexes]

    #convert numpy to tensor
    self.data = convert_to_tensor(self.data)
    
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):#load data on demand
    return self.data[idx], self.labels[idx]
