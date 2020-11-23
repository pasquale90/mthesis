#import libraries
import numpy as np
import pandas as pd
import librosa
from PIL import Image
import torch
from torchvision import transforms


#AUDIO AUGMENTATION FUNCTIONS_ synthetic data
def audio_augmentation(data, sr, class_conditional, thresshold):

  #add_white_noise to the signal
  def white_noise(data): 
    noiz = np.random.randn(len(data))
    mean_intensity = np.sum(np.square(data))/len(data)
    data_wn = data + noiz*0.75* mean_intensity
    return data_wn
    
  #Shift the sound wave by a factor value chosen randomly within [0.5,1,1.5,2] seconds
  def time_shift(data,sr):
    time_step = np.random.choice([sr//2,sr,sr+sr//2,sr*2])
    time_shifted = np.roll(data,time_step)
    return time_shifted

  #Time-stretching the wave by a factor value of 0.9. Permissible : 0 < x < 1.0
  def time_stretch(data):#,factor
    factor = 0.90
    time_streched = librosa.effects.time_stretch(data,factor)
    return time_streched[0:len(data)]

  #pitch shifting of wave by a random factor value in the space [-1,1].  Permissible : -5 <= x <= 5
  def pitch_shift(data,sr):
    detune = np.random.uniform(low=-2.5,high=-1.75,size=None)
    overtune = np.random.uniform(low=1.75,high=2.5,size=None)
    shift_factor = np.random.choice([detune,overtune])
    pitch_shifted = librosa.effects.pitch_shift(data,sr,n_steps=shift_factor)
    return pitch_shifted
  
  def soft_pitch_shift(data,sr):
    detune = np.random.uniform(low=-1.0,high=-0.5,size=None)
    overtune = np.random.uniform(low=0.5,high=1.0,size=None)
    shift_factor = np.random.choice([detune,overtune])
    pitch_shifted = librosa.effects.pitch_shift(data,sr,n_steps=shift_factor)
    return pitch_shifted
  
  strong_augs = ['airplane','car_horn','cat',#esc
                 'chirping_birds','church_bells',
                 'cow','crow','crying_baby',
                 'door_wood_creaks','insects',
                 'rooster','sheep','siren',
                 'car_horn','children_playing','siren']#us8k

  medium_augs= ['breathing','brushing_teeth',#esc
                'clock_alarm','coughing','dog',
                'door_wood_knock','fireworks',
                'frog','glass_breaking','hand_saw',
                'hen','laughing','pig','pouring_water',
                'sneezing','snoring',
                'dog_bark','gun_shot','street_music']#us8k
                
  weak_augs = ['can_opening','chainsaw','clapping',#esc
               'clock_tick','crackling_fire','crickets',
               'drinking_sipping','engine','footsteps',
               'helicopter','keyboard_typing','mouse_click',
               'rain','sea_waves','thunderstorm',
              'toilet_flush','train','vacuum_cleaner',
              'washing_machine','water_drops','wind',  
               'air_conditioner','drilling','engine_idling','jackhammer']#us8k
   
  #prob_wn = np.random.uniform(low=0,high=1)
  #if prob_wn>thresshold:
  data =  white_noise(data)
  
  #prob_tsh = np.random.uniform(low=0,high=1)
  #if prob_tsh>thresshold:
  data = time_shift(data,sr)

  if class_conditional in strong_augs:
    
    prob_tst = np.random.uniform(low=0,high=1)
    if prob_tst>thresshold:
      data = time_stretch(data)

    prob_psh = np.random.uniform(low=0,high=1)
    if prob_psh>thresshold:
      data = pitch_shift(data,sr)
  
  elif class_conditional in medium_augs:
    
    prob_spsh = np.random.uniform(low=0,high=1)
    if prob_spsh>thresshold:
      data = soft_pitch_shift(data,sr)

  elif class_conditional in weak_augs:
    pass

  return data

# Data augmentation and normalization for training
# Just normalization for validation
train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale = (0.05,0.05), ratio = (0.3,0.33), value=0, inplace=False)
        
    ])
valid_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

#Vision Augmentations -  Horizontal Flip, Random Erasing
def vision_augmentations(data,transform):
    augs = []
    for file in data:
      image = Image.fromarray(file)
      image = transform(image)
      augs.append(image)
    return augs
