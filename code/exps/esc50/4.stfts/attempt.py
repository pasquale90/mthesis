#keep a track of experiment's attempts in order to avoid overwritting issues
import os

import paths


class attempt_class:
  #init_attempt
  def __init__(self,mode,vfold):
    self.attempt_path = paths.attempt_path
    self.attempt_file = paths.attempt_path+"attempts_"+mode+"_"+str(vfold)+".txt"
    if (not os.path.exists(self.attempt_path)):
      os.mkdir(self.attempt_path)
      self.init_files()
    elif (not os.path.isfile(self.attempt_file)):
      self.init_files()
    else:
      print(f'attempt is already initialized')
      print(f'Experiment\'s _{mode} attempt no_ : {self.get_attempt()}')
  
  def init_files(self):
    attempt = 1
    f = open(self.attempt_file,'w')
    with open(self.attempt_file, 'a') as out:
      out.write(str(attempt))
      print(f'Experiment\'s attempt no_ : {attempt}')
  
  def add_attempt(self):
    with open(self.attempt_file, "r") as f:
      attempt = f.read()
      attempt = int(attempt)
      attempt+=1
    with open(self.attempt_file, "w") as f:
      f.write(str(attempt))
    print(f'Experiment\'s attempt changed to : {attempt}')

  def get_attempt(self):
    with open(self.attempt_file, "r") as f:
      attempt = int(f.read())
    return attempt

  def set_attempt(self,attempt_value):
    with open(self.attempt_file, "w") as f:
      f.write(str(attempt_value))
    print(f'Experiment\'s attempt is set to : {attempt_value}')
