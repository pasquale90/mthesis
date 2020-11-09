class prevent_overfitting:
  def __init__(self):
    self.tolerance=5 #3epochs tolerance
    self.minloss = 10 
    self.lossenvelope =[]
    
    self.thresshold = 0#0.5
    self.avgperformances=[]
    self.best_epoch = 0
    self.micro_accuracies = []
    self.macro_accuracies = []
    self.best_curr_model = False#None #model.state_dict save
  
  #check overfitting by observing 3 last epoch's valid loss
  def detect_overfitting(self,validloss,epoch):
    self.lossenvelope.append(validloss)
    if validloss<self.minloss:
      self.minloss=validloss
    print('minloss',self.minloss)

    if (len(self.lossenvelope)>self.tolerance):
      self.lossenvelope.pop(0) #reject first value when 3 values are passed

    if(len(self.lossenvelope)>=self.tolerance):
      overfit=all(earlier <= later for earlier, later in zip(self.lossenvelope, self.lossenvelope[-4:]))#check if descending
      if (overfit and min(self.lossenvelope)>self.minloss):
        return True
    return False
  
  #store best model's results by observing mean micro and macro accuracies
  def store_best_model(self,micro_accuracy,macro_accuracy):
    self.macro_accuracies.append(micro_accuracy)
    self.micro_accuracies.append(macro_accuracy)
    
    meanaccuracy = (micro_accuracy+macro_accuracy)/2.0
    self.avgperformances.append(meanaccuracy)
    
    self.best_epoch = self.avgperformances.index(max(self.avgperformances))+1
    
    if meanaccuracy >= self.thresshold:
      self.best_curr_model = (meanaccuracy >= max(self.avgperformances))
      print(f'just saved the best current model in epoch{self.best_epoch}, with acc1:{self.micro_accuracies[self.best_epoch-1]}, and acc2:{self.macro_accuracies[self.best_epoch-1]}')
    else:
      self.best_curr_model = False

    return self.best_curr_model, self.micro_accuracies[self.best_epoch-1],self.macro_accuracies[self.best_epoch-1]#, self.best_epoch
  
  def early_stopping(self,epochinst):
    print('training is terminating so as to prevent further overfitting')
    total_epochs = epochinst.set_total(epochinst.get_step())
    return total_epochs


class epochs_class:
  def __init__(self):
    self.total_epochs=50
    self.step_epoch=1
  def set_total(self,num_epochs):
    self.total_epochs=num_epochs
    return self.get_total()
  def get_total(self):
    return self.total_epochs
  def next_step(self):
    self.step_epoch+=1
    return self.step_epoch
  def get_step(self):
    return self.step_epoch
