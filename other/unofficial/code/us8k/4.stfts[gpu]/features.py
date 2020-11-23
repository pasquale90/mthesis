import numpy as np
import pandas as pd
import librosa

import augmentation

#define analysis parameters
def analysis_parameters(mode):
  if mode == '1':#257x251---->choice 1
    sampling_rate=16000
    hop_length=256
    fft_points=512
  elif mode=='2':#513x_345
    sampling_rate=44100
    hop_length=512
    fft_points=1024
  elif mode == '3' : #257x_
    sampling_rate=22050
    hop_length=256
    fft_points=512

  return sampling_rate, hop_length, fft_points
  
def zero_pad(signal,fs):
  shift_time = True
  if len(signal)>(4*fs):
    signal = signal[0:4*fs]
  elif len(signal)<(4*fs):
    shift_time = False
  num_zeros=4*fs-len(signal)
  zp=np.zeros(num_zeros,dtype=float)
  padded_signal = np.concatenate((signal,zp),axis=0)
  return padded_signal, shift_time

#extract features
def extract_stft_spectogram(audio_path, df, folds, audiofiles, sr, hop, nfft):   

  def compute_stft_spectogram(raw,sr,hop,nfft,window='hann'):
    stft=librosa.core.stft(y=raw.astype(float),
                            n_fft=nfft, 
                            hop_length=hop, 
                            window='hann',
                            center=True)
    magnitude = np.abs(stft)
    #to spectogram 
    S = librosa.amplitude_to_db(magnitude, ref=np.max )
    return S

  def scale_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled
  
  features, labels, folders = [], [], []  
 
  extr = True
  if extr == True:

    print('Extracting features ........ ')
    
    shape_print = True
    pad_print = True

    #deterministic random augmentation
    np.random.seed(77)

    for foldname, filesinfold in audiofiles.items():
      
     #if foldname=='fold9' or foldname=='fold10':#test

      path = audio_path+foldname+'/'

      for file in filesinfold:						#[0:200]: test
        name = file.split('.wav')[0]
        label = np.int8(file.split('-')[1])
        folder = np.int8(folds.index(foldname)+1)
        #print('name', name)
        #print('folder',folder)
        #print('label',label) 

        raw,_ = librosa.load(path+file, sr=sr, mono=True)
      
        #zero pad signal to 4 seconds
        padded, shift_time = zero_pad(raw,sr)

        #extract mel spectogram
        S = compute_stft_spectogram(padded,sr,hop,nfft)

        #flip image
        flipped = np.flipud(S)

        #to gray scale
        greyscale = scale_image(flipped)
      
        features.append(greyscale)
        labels.append(label)
        folders.append(folder)

        #Synthetic data augmentations
        prob = np.random.uniform(low=0,high=1)
        if prob<=1:#10.5:
          category = df.loc[df['slice_file_name']==file]['Class'].to_string(index=False).lstrip()
          augmented = augmentation.audio_augmentation(data=padded,sr=sr,class_conditional=category,shift_time=shift_time,thresshold=0.5)
          synthetic = scale_image(np.flipud(compute_stft_spectogram(augmented,sr,hop,nfft)))

          features.append(synthetic)
          labels.append(label)
          folders.append(folder)

        #test_shapes of raw data and feature representation
        if shape_print:
          print('\nFeature Shape Check\n')
          print(f'raw had len:{len(raw)/sr}, and padded has len:{len(padded)/sr}')
          print(f'Spectogram has shape : {S.shape} with min:{greyscale.min()} and max:{greyscale.max()}]')
          shape_print = False
        if (not shift_time) and pad_print:
          print('\nPadded Feature Shape Check\n')
          print(f'raw had len:{len(raw)/sr}, and padded has len:{len(padded)/sr}')
          print(f'Spectogram has shape : {S.shape} with min:{greyscale.min()} and max:{greyscale.max()}')
          pad_print = False

  '''
  print('len(features)-features',len(features))
  print('len(features[0])-freq_domain',len(features[0]))
  print('len(features[0][0])-time_domain',len(features[0][0]))
  print('labels',len(labels))
  print('folders',len(folders))
  '''
  
  print('Features are extracted!')

  return features, labels, folders

