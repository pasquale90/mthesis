import pandas as pd
import numpy as np
import os

import paths

#save results
def save_results(content, name):
  path = paths.results_path
  if (not os.path.exists(path)):
    os.mkdir(path)

  ovr_results, class_results = content
  ovr_results_filename, class_results_filename = name 
  
  ovr_results.to_csv(path+ovr_results_filename) 
  class_results.to_csv(path+class_results_filename) 
   

#func to define the last argument in the save_results method
#PATTERN:dataset&exp_mode_validationfold_attempt_metrics.csv
def define_filenames_pattern(expid, mode, vfold, attempt):
  classf1_report_filename = expid+'_'+mode+'_v'+str(vfold)+'_a'+str(attempt)+'_classF1.csv'
  ovr_results_filename=expid+'_'+mode+'_v'+str(vfold)+'_a'+str(attempt)+'_overalF1.csv'
  return ovr_results_filename , classf1_report_filename 

#as the first argument in save_results
def define_content(epochs,
		   mean_train_losses, 
                   mean_valid_losses, 
                   microf1, 
                   macrof1, 
		   classf1):
                   
  #data = [mean_train_losses, mean_valid_losses, accuracies,micro_auroc, macro_auroc microf1, macrof1]#auroc,

  data = {'mean_train_loss' : mean_train_losses, 'mean_valid_loss' : mean_valid_losses,
	 'micro_f1' : microf1, 'macro_f1' : macrof1 }
  
  #index names
  epochs_index= (['epoch_'+str(ep+1) for ep in range(epochs)])
  
  overal_results = pd.DataFrame(data=data, index=epochs_index,  dtype=np.float16)
  class_results = pd.DataFrame(data=classf1, index=epochs_index,  dtype=np.float16)

  return overal_results, class_results

#save general results so as to compare with other systems in a different folder
def save_genres(micro, macro, params, validation_fold, filename):
  
  path = paths.compare_results_path

  #if csv exists, overwrite results
  if os.path.isfile(path+filename):
    general_results=pd.read_csv(path+filename)
    general_results.set_index('validation_fold:',inplace=True)
    new_fold_results = [micro, macro, params]
    general_results[validation_fold] = new_fold_results
    general_results.to_csv(path+filename)
  #if not, store them into a new csv file
  else:
    data = {validation_fold : [micro,macro,params]}
    general_results = pd.DataFrame(data=data, index=['micro_f1','macro_f1','params'], dtype=np.float16)
    general_results.index.name = 'validation_fold:'
    general_results.to_csv(path+filename) 
    print(general_results)

def genres_filename(expid,mode):
  filename=expid+'_'+str(mode)+'.csv'
  return filename
