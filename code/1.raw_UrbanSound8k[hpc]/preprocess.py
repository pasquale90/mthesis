import numpy as np
import pandas as pd
import librosa

import augmentation

#define analysis parameters
def analysis_parameters(mode):
  sampling_rate=int(mode)*1000  #for instance: if mode 16, sr = 16kHz
  
  if mode=='16':
    #sampling_rate=16000
    window_size = sampling_rate
    overlap = int(window_size*0.75)
    hop_length = (window_size-overlap)
  elif mode=='32':
    #sampling_rate=32000
    window_size = sampling_rate
    overlap = int(window_size*0.5)
    hop_length = (window_size-overlap)
  return sampling_rate, window_size, hop_length


def preprocess_data(audio_path, df, folds, audiofiles, mode):

  '''
  def zero_pad_to_closest(signal,fs):
    shift_time = True
    if len(signal)>(4*fs):
      signal = signal[0:4*fs]
    elif len(signal)<(4*fs):
      shift_time = False
      length = len(signal)
      num_zeros = (((round(length/fs))+1)*fs)-length 
      zp=np.zeros(num_zeros,dtype=float)
      signal = np.concatenate((signal,zp),axis=0)
    return signal, shift_time
  '''

  def zero_pad(signal,fs):
    shift_time = True
    if len(signal)>(4*fs):
      signal = signal[0:4*fs]
    elif len(signal)<(4*fs):
      shift_time = False
      num_zeros=4*fs-len(signal)  
      zp=np.zeros(num_zeros,dtype=float)
      signal = np.concatenate((signal,zp),axis=0)
    return signal, shift_time

  def normalize(data):#in the space [-1,1]
    #return (data - np.min(data)) / (np.max(data) - np.min(data))#[0,1]
    return (2*(data-np.min(data))/(np.max(data) - np.min(data)))-1

  def _windows(data, window_size, hop_length):
    start = 0
    while start < len(data):
      yield start, start + window_size
      start += hop_length
  
  def flatten_features(feature):
    feature = np.asarray(feature,dtype=np.float32).flatten()
    return feature.tolist()
  
  features, labels, folders = [], [], []  
 
  extr = True
  if extr == True:

    print('Preprocessing data ........ ')
    
    sampling_rate, window_size, hop_length = analysis_parameters(mode)
    #print(f'sampling_rate: {sampling_rate}, window_size: {window_size}, hop_length: {hop_length}')

    shape_print = True
    pad_print = True

    #deterministic random augmentation
    np.random.seed(77)

    for foldname, filesinfold in audiofiles.items():
      
     #if foldname=='fold2' or foldname=='fold1':#test

      path = audio_path+foldname+'/'

      for file in filesinfold:						#[0:200]: test

        name = file.split('.wav')[0]
        label = np.int8(file.split('-')[1])
        folder = np.int8(folds.index(foldname)+1)
        #print('name', name)
        #print('folder',folder)
        #print('label',label) 

        raw,_ = librosa.load(path+file, sr=sampling_rate, mono=True)
        #print(f'{file} had length {len(raw)}')

        #normalize
        raw = normalize(raw)

        #zero pad signal to 4 seconds
        padded, shift_time = zero_pad(raw,sampling_rate)
        #print(f'now has length {len(padded)}')

        frames = []
        #get windows out of the raw waveform
        for count_frames,(start,end) in enumerate(_windows(padded,window_size,hop_length)):

          if(len(padded[start:end]) == window_size):
            #print(start,end)
            frame = padded[start:end]#rectangular window

            frames.append(frame)
            
        #flatten
        #flatten implemented within the model architecture
        
        features.append(frames)
        labels.append(label)
        folders.append(folder)
        
        #Synthetic data augmentations
        prob = np.random.uniform(low=0,high=1)
        if prob<=1:#10.5:
          category = df.loc[df['slice_file_name']==file]['Class'].to_string(index=False).lstrip()
          augmented = augmentation.audio_augmentation(data=padded,sr=sampling_rate,class_conditional=category,shift_time=shift_time,thresshold=0.5)
          synth_frames = []
          for count_frames,(start,end) in enumerate(_windows(augmented,window_size,hop_length)):

            if(len(augmented[start:end]) == window_size):
              #print('AUGMENTED',start,end)
              synthetic_frame = augmented[start:end]#rectangular window

              synth_frames.append(synthetic_frame)

          features.append(synth_frames)
          labels.append(label)
          folders.append(folder)

         
        
          #test_shapes of raw data and feature representation
          if shape_print:
            print('\nFeature Shape Check\n')
            print(f'raw had len:{len(raw)/sampling_rate}, and padded has len:{len(padded)/sampling_rate}')
            print(f'Postprocessed feature has shape : {np.asarray(flatten_features(frames)).shape} with min:{np.asarray(frame).min()} and max:{np.asarray(frame).max()}]')
            shape_print = False
          if (not shift_time) and pad_print:
            print('\nPadded Feature Shape Check\n')
            print(f'raw had len:{len(raw)/sampling_rate}, and padded has len:{len(padded)/sampling_rate}')
            print(f'Postprocessed feature has shape : {np.asarray(synthetic_frame).shape} with min:{np.asarray(synthetic_frame).min()} and max:{np.asarray(synthetic_frame).max()}]')
            pad_print = False
        
  '''
  print('len(features)-features',len(features))
  print('len(features[0])-freq_domain',len(features[0]))
  print('len(features[0][0])-time_domain',len(features[0][0]))
  print('labels',len(labels))
  print('folders',len(folders))
  '''
  
  #print('Features are extracted!')

  return features, labels, folders
