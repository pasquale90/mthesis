import numpy as np
import pandas as pd
import librosa

import augmentation

#define analysis parameters
def analysis_parameters(mode):
  if mode == '1':	#--->choice 1
    sampling_rate=16000
    hop_length=256
    fft_points=512
  elif mode=='2':
    sampling_rate=44100
    hop_length=512
    fft_points=1024
  elif mode == '3' :
    sampling_rate=22050
    hop_length=256
    fft_points=512

  return sampling_rate, hop_length, fft_points

#extract features
def extract_stft_spectogram(audio_path, df, audiofiles, sr, hop, nfft):   

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

    shape_print=True
    
    for count_files, file in enumerate(audiofiles):
      name = file.split('.wav')[0]
      label = int(file.split('-')[-1].split('.')[0])
      folder = int(file.split('-')[0])
      #print('name', name)
      #print('folder',folder)
      #print('label',label) 


      raw,_ = librosa.load(audio_path+file, sr=sr, mono=True)
      S = compute_stft_spectogram(raw,sr,hop,nfft)

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
      synthetic = scale_image(np.flipud(compute_stft_spectogram(augmented,sr,hop,nfft)))

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
