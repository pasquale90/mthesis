import os
import pandas as pd
import numpy as np
import scipy as sp
from scipy.io import wavfile
import shutil
#read annotations_file
annotations = 'C:/Users/Alex/Desktop/Πασχάλης/annotations.csv'
raw_annotations = pd.read_csv(annotations)

train_split = raw_annotations[raw_annotations.split == 'train']
validation_split = raw_annotations[raw_annotations.split != 'train']
ground_truth = raw_annotations[raw_annotations.annotator_id == 0]

#check splits
train_split= train_split['audio_filename'].unique()
validation_split= validation_split['audio_filename'].unique()
ground_truth=ground_truth['audio_filename']

train_split=pd.Series(train_split)
validation_split=pd.Series(validation_split)

#Sort by indexes
train_split.reset_index(inplace=True, drop=True )
validation_split.reset_index(inplace=True, drop=True )
ground_truth.reset_index(inplace=True, drop=True )

print ('train_split_shape:', train_split.shape)
print ('validation_split_shape:', validation_split.shape)
print ('ground_truth_shape:',ground_truth.shape)

print ('train_split_type:', type(train_split))#SIZE=13538
print ('validation_split_type:', type(validation_split))#SIZE=4308
print ('ground_truth_type:',type(ground_truth))

#audiofiles 
audio_path=(r'C:\Users\Alex\Desktop\Πασχάλης\audio')
audio_files=os.listdir(audio_path)
print(len(audio_files))
print(audio_files[0])

#def getpath
ground_truth_path=(r'C:\Users\Alex\Desktop\Πασχάλης\ground_truth')
validation_path=(r'C:\Users\Alex\Desktop\Πασχάλης\validation_split')
train_path=(r'C:\Users\Alex\Desktop\Πασχάλης\train_split')

#Move files
#def movefiles()
for afn in train_split:
    print("afn:\t",afn)
    shutil.move(audio_path+"/"+afn, train_path)
