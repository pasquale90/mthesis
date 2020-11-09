import os
import paths

def makefile(mode,vfold):
  console_path = paths.console_path
  console_file = paths.console_path+"console_"+mode+"_"+str(vfold)+".txt"
  if (not os.path.exists(console_path)):
      os.mkdir(console_path)
      statement = mode+"_"+str(vfold)
      f = open(console_file,'w')
      with open(console_file, 'a') as out:
        out.write(statement)
  elif (not os.path.isfile(console_file)):
      statement = mode+"_"+str(vfold)
      f = open(console_file,'w')
      with open(console_file, 'a') as out:
        out.write(statement)
  
  return console_file
