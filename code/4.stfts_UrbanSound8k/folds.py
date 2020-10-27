import os

def read_folds(audio_path,folds):
  folds=os.listdir(audio_path)
  
  order={}
  files={}
  filesum=0
  
  for f in folds:
  
    filelist = os.listdir(audio_path+f+'/')
   
    if (f[-1:]!='0'):
      num=f[-1:]
    elif(f[-1:]=='0'):#get the value of 10 instead of 0
      num=f[-2:]
  
    order[f]=int(num)
    files[f]=filelist

    filesum+=len(files[f])
    
    
    
  folds={}
  for key, value in sorted(order.items(), key=lambda item: item[1]):
    folds[value]= key
  folds=list(folds.values())
  
  return folds, files, order, filesum

