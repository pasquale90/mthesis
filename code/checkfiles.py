import os
from PIL import Image
import numpy as np
import pandas as pd


path80="/mnt/scratch_a/users/m/melissap/features/ur8k/mel/80/"
path128="/mnt/scratch_a/users/m/melissap/features/ur8k/mel/128/"
pathe4="/mnt/scratch_a/users/m/melissap/features/ur8k/stfts/exp4/"
pathe8="/mnt/scratch_a/users/m/melissap/features/ur8k/stfts/exp8/"

files_80=os.listdir(path80)
files_128=os.listdir(path128)
files_e4=os.listdir(pathe4)
files_e8=os.listdir(pathe8)

def read_image(path,name):
  image=Image.open(path+name)
  #display(image)
  image=np.array(image) 
  return image

min80=1000
max80=0
min128=1000
max128=0
mine4=1000
maxe4=0
mine8=1000
maxe8=0

for f in range(len(files_80)):
  width80 = read_image(path80,files_80[f]).shape[1]
  width128 = read_image(path128,files_128[f]).shape[1]
  widthe4 = read_image(pathe4,files_e4[f]).shape[1]
  widthe8 = read_image(pathe8,files_e8[f]).shape[1]

  if (width80<min80):
    min80=width80
  elif(width80>max80):
    max80=width80
  if (width128<min128):
    min128=width128
  elif(width128>max128):
    max128=width128
  if (widthe4<mine4):
    mine4=widthe4
  elif(widthe4>maxe4):
    maxe4=widthe4
  if (widthe8<mine8):
    mine8=widthe8
  elif(widthe8>maxe8):
    maxe8=widthe8

print(f'80\t:min-->{min80},max-->{max80}')
print(f'128\t:min-->{min128},max-->{max128}')
print(f'e4\t:min-->{mine4},max-->{maxe4}')
print(f'e8\t:min-->{mine8},max-->{maxe8}')

