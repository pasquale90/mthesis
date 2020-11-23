import numpy as np
import pandas as pd
import librosa

import augmentation

#define analysis parameters
def analysis_parameters(mode):
  sampling_rate=44100
  hop_length=512
  fft_points=2048
  mel_bands=mode  #80x431 or 128x431
  return sampling_rate, hop_length, fft_points, mel_bands
  
#extract features
def extract_mel_spectogram(audio_path, df, audiofiles, sr, hop, nfft, nmels):   

  def compute_mel_spectogram(raw,sr,hop,nfft,nmels,window='hann'):
    S=librosa.feature.melspectrogram(y=raw.astype(float),
                                      sr=sr,S=None,
                                      n_fft=nfft,
                                      hop_length=hop,
                                      window=window, 
                                      power=2,
                                      n_mels=nmels) 
    S = librosa.power_to_db(S, ref=np.max)
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

    shape_print=True
    
    for count_files, file in enumerate(audiofiles):
      name = file.split('.wav')[0]
      label = int(file.split('-')[-1].split('.')[0])
      folder = int(file.split('-')[0])
      #print('name', name)
      #print('folder',folder)
      #print('label',label) 


      raw,_ = librosa.load(audio_path+file, sr=sr, mono=True)
      S = compute_mel_spectogram(raw,sr,hop,nfft,nmels)

      #flip image
      flipped = np.flipud(S)

      #to gray scale
      greyscale = scale_image(flipped)
      
      features.append(greyscale)
      labels.append(label)
      folders.append(folder)

      #Synthetic data augmentations
      category = df.loc[df['filename']==file]['category'].to_string(index=False).lstrip()
      augmented = augmentation.audio_augmentation(data=raw,sr=sr,class_conditional=category,thresshold=0.5)
      synthetic = scale_image(np.flipud(compute_mel_spectogram(augmented,sr,hop,nfft,nmels)))

      features.append(synthetic)
      labels.append(label)
      folders.append(folder)
      
      if shape_print:
          print('\nFeature Shape Check\n')
          print(f'Spectogram has shape : {S.shape} with min:{greyscale.min()} and max:{greyscale.max()}')
          shape_print = False


  '''
  print('len(features)-features',len(features))
  print('len(features[0])-freq_domain',len(features[0]))
  print('len(features[0][0])-time_domain',len(features[0][0]))
  print('labels',len(labels))
  print('folders',len(folders))
  '''
  
  print('Features are extracted!')

  return features, labels, folders
