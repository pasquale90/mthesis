import numpy as np
import pandas as pd
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import classification_report

def F1_score(trace_y, trace_yhat, esc_classes):
  num_classes = len(esc_classes)
  #confussion matrix
  confmat=pd.DataFrame(data=np.zeros(shape=(num_classes,num_classes)))
  predictions=trace_yhat.argmax(axis=1)

  for i,pred in enumerate(predictions):
    confmat.iat[trace_y[i],pred]+=1
    
  TP=pd.Series(confmat.to_numpy().diagonal())
  FP=confmat.sum(axis=0)-TP
  FN=confmat.sum(axis=1)-TP

  TN=confmat.sum().sum()-TP-FP-FN
  
  #micro
  micro_precision = TP.sum()/(TP.sum()+FP.sum())
  micro_recall = TP.sum()/(TP.sum()+FN.sum())
  F1_micro=2*micro_precision*micro_recall/(micro_precision+micro_recall)
  #macro
  macro_precision=TP/(TP+FP)
  macro_precision.fillna(value=0.0)

  macro_recall=TP/(TP+FN)
  macro_recall.fillna(value=0.0)

  F1_macro=2*macro_precision*macro_recall/(macro_precision+macro_recall)
  F1_macro.replace(np.nan, 0, inplace=True)
  F1_macro=np.mean(F1_macro)

  return F1_micro,F1_macro
  
#labels, preds
def confusion_matrix(trace_y,trace_yhat, num_classes):
  confmat=pd.DataFrame(data=np.zeros(shape=(num_classes,num_classes)))
  predictions=trace_yhat.argmax(axis=1)

  for i,pred in enumerate(predictions):
    confmat.iat[trace_y[i],pred]+=1

  return confmat


def F1_Class(trace_y,trace_yhat,esc_classes):
  num_classes = len(esc_classes)
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
    if (FP[i]==0):
      class_precision[i]=1
    else:
      class_precision[i] = TP[i]/(TP[i]+FP[i]) 
    if (FN[i]==0):
      class_recall[i]=1
    else:
      class_recall[i] = TP[i]/(TP[i]+FN[i])
    
    if (class_precision[i]+class_recall[i]==0.0):
      class_f1[i]=0
    else:
      class_f1[i] = 2*class_precision[i]*class_recall[i]/(class_precision[i]+class_recall[i])
    
  #create a dict_report
  f1_class_report = {}
  class_report = {}
  class_counts = np.asarray(np.unique(trace_y, return_counts=True)).T

  for i,c in enumerate(esc_classes):
    f1_class_report[c] = {}
    f1_class_report[c]['precision'] = class_precision[i]
    f1_class_report[c]['recall'] = class_recall[i]
    f1_class_report[c]['f1'] = class_f1[i]
    f1_class_report[c]['count'] = class_counts[i][1]

  return f1_class_report

'''
def get_accuracy(trace_yhat, trace_y):
  accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
  return accuracy



#area under roc curve
def avg_auroc(trace_y,trace_yhat):
  shaped_y=[]
  for val in range(len(trace_y)):
    templine=[]
    for t in range(50):
      if(t==trace_y[val]):
        templine.append(1)
      else:
        templine.append(0)
    shaped_y.append(templine) 
  micro_auroc = roc_auc_score(shaped_y, trace_yhat, multi_class="ovo",
                                  average="micro")

  macro_auroc = roc_auc_score(shaped_y, trace_yhat, multi_class="ovo",
                                  average="macro")

  return micro_auroc, macro_auroc 


def F1_per_Class(trace_y,trace_yhat, classes):
  f1_classreport = classification_report(trace_y, trace_yhat.argmax(axis=1), target_names=classes, output_dict=True)
  return f1_classreport
'''
