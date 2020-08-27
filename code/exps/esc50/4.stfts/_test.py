import numpy as np
import pandas as pd
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import optimizer
import model
import torch

import optimizer
from utils import epochs_class,prevent_overfitting

#define device
if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')
print(device)

mode='128'
model = model.esc_mel_model_hybrid(input_shape=(1,128,431), batch_size=16, num_cats=50, mode=mode).to(device)
print('model initialized...')

#define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
learning_rate = 2e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochinst=epochs_class()
total=epochinst.set_total(50)
current=epochinst.get_step()

valid_loss=[0.99146676, 0.98174042, 0.93956269 ,0.938171,    0.91415392,0.93703358,
 0.90373795, 0.86014409, 0.83936784, 0.82996061,  0.73301917,0.81359223,
 0.71572811, 0.7065203,  0.67730309, 0.66746673, 0.66180608, 0.57310721,
 0.56848232, 0.51053962, 0.49265659, 0.4916381 , 0.45557463, 0.45115155,
 0.40003451, 0.38943551, 0.37583768, 0.36961226, 0.35231847, 0.29474258,
 0.26424613, 0.24177685, 0.23957188, 0.20694761, 0.18858694, 0.18612436,
 0.18365473, 0.17283884, 0.16000254, 0.15036157,  0.1393833,0.13967004,
 0.09514894, 0.09236013, 0.08423445, 0.08349973,  0.05289081,0.05372718,
 0.03821884, 0.03342239]

overfit=prevent_overfitting()

while current<=total:  #while
  print(current)
  early_stop,optimizer = overfit.detect_overfitting(valid_loss[current-1],optimizer, learning_rate, epochinst)
      
  if early_stop:
    total=epochinst.set_total(epochinst.get_step())
  current=epochinst.next_step()#step
