import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

#define metrics
def F1_score(trace_y, trace_yhat, classes):
  num_classes = len(classes)
  #confussion matrix
  confmat = confusion_matrix(trace_y, trace_yhat, num_classes)
    
  TP=pd.Series(confmat.to_numpy().diagonal())
  FP=confmat.sum(axis=0)-TP
  FN=confmat.sum(axis=1)-TP

  TN=confmat.sum().sum()-TP-FP-FN
  
  #micro
  micro_precision = TP.sum()/(TP.sum()+FP.sum())
  micro_recall = TP.sum()/(TP.sum()+FN.sum())
  F1_micro=2*micro_precision*micro_recall/(micro_precision+micro_recall)

  #macro
  macro_precision = pd.Series(np.nan)
  macro_recall= pd.Series(np.nan)
  macro_f1 = pd.Series(np.nan)
  #Avoid Zero-Division
  for i in range(num_classes):
    if (TP[i]==0 and FP[i]==0):
      macro_precision[i]=0
    else:
      macro_precision[i] = TP[i]/(TP[i]+FP[i]) 
    if (TP[i]==0 and FN[i]==0):
      macro_recall[i]=0
    else:
      macro_recall[i] = TP[i]/(TP[i]+FN[i])
    
    if (macro_precision[i]+macro_recall[i]==0.0):
      macro_f1[i]=0
    else:
      macro_f1[i] = 2*macro_precision[i]*macro_recall[i]/(macro_precision[i]+macro_recall[i])
      
  macro_precision = macro_precision.sum()/num_classes
  macro_recall=macro_recall.sum()/num_classes
  F1_macro = macro_f1.sum()/num_classes

  return micro_recall,micro_precision,F1_micro, macro_recall,macro_precision,F1_macro
  
#labels, preds
def confusion_matrix(trace_y,trace_yhat, num_classes):
  confmat=pd.DataFrame(data=np.zeros(shape=(num_classes,num_classes)))
  predictions=trace_yhat.argmax(axis=1)

  for i,pred in enumerate(predictions):
    confmat.iat[trace_y[i],pred]+=1

  return confmat


def F1_Class(trace_y,trace_yhat,classes):
  num_classes = len(classes)
  confmat = confusion_matrix(trace_y, trace_yhat, num_classes)
  
  TP=pd.Series(confmat.to_numpy().diagonal())
  FP=confmat.sum(axis=0)-TP
  FN=confmat.sum(axis=1)-TP
  TN=confmat.sum().sum()-TP-FP-FN

  #AVOID ZERO DIVISION
  class_precision = pd.Series(np.nan)
  class_recall = pd.Series(np.nan)
  class_f1 = pd.Series(np.nan)
  for i in range(num_classes):
    if (TP[i]==0 and FP[i]==0):
      class_precision[i]=0
    else:
      class_precision[i] = TP[i]/(TP[i]+FP[i]) 
    if (TP[i]==0 and FN[i]==0):
      class_recall[i]=0
    else:
      class_recall[i] = TP[i]/(TP[i]+FN[i])
    
    if (class_precision[i]==0.0 and class_recall[i]==0.0):
      class_f1[i]=0
    else:
      class_f1[i] = 2*class_precision[i]*class_recall[i]/(class_precision[i]+class_recall[i])
    
  #create a dict_report
  f1_class_report = {}
  class_report = {}
  class_counts = np.asarray(np.unique(trace_y, return_counts=True)).T

  for i,c in enumerate(classes):
    f1_class_report[c] = {}
    f1_class_report[c]['precision'] = class_precision[i]
    f1_class_report[c]['recall'] = class_recall[i]
    f1_class_report[c]['f1'] = class_f1[i]
    f1_class_report[c]['count'] = class_counts[i][1]

  return f1_class_report


from sklearn.metrics import classification_report
import sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

def backup_metrics(labels,preds,classes,path,mode,vfold):
    if (not os.path.exists(path)):
      os.makedirs(path)
    report = classification_report(labels, preds.argmax(1), target_names=classes)
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-5]:
        row = {}
        row_data = ' '.join(line.split())   
        row_data = row_data.split(' ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(path+'backup_classification_report_'+str(mode)+'_'+str(vfold)+'.csv',index=False)

    accuracy= f1_score(labels, preds.argmax(1), average='micro', zero_division='warn')
    macro_avg= f1_score(labels, preds.argmax(1), average='macro', zero_division='warn')
    weighted = f1_score(labels, preds.argmax(1), average='weighted', zero_division='warn')
    precision,recall,_,_ =precision_recall_fscore_support(labels, preds.argmax(1), average='macro')
    general = pd.DataFrame(data = [accuracy,recall,precision,macro_avg,weighted],index=['accuracy','recall','precision','macro_f1','weighted_f1'])
    general.to_csv(path+'backup_general_'+str(mode)+'_'+str(vfold)+'.csv')
