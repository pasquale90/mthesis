from optimizer import reduce_lr

class prevent_overfitting:
  def __init__(self):
    self.flag_1=False#detect raise in error _ do nothing
    self.flag_2=False#detect raise in error _ do nothing
    self.flag_3=False#detect raise in error _ early_stopping
    self.performances=[]
  def detect_overfitting(self,validloss,optimizer,lr,epochinst):
    self.performances.append(validloss)#append new value
    if (len(self.performances)>2):
      self.performances.pop(0) #reject first value when 3 values are passed
    if(len(self.performances)>=2):
      overfit=all(earlier <= later for earlier, later in zip(self.performances, self.performances[1:]))#check if descending
      #print(f'overfitvar:{overfit}')
      if (overfit):#if true ->then overfitting --> reduce lr, flag1. flag2-->set epoch=epoch to stop training
        print ('Overfitting detected')
        if (self.flag_1==False):
          self.flag_1=True
        elif(self.flag_1==True):
          if (self.flag_2==False):
            optimizer = reduce_lr(optimizer,lr)
            print('learning rate reduced..')
            self.flag_2=True
          elif(self.flag_2==True):
            if (self.flag_3==False):
              optimizer = reduce_lr(optimizer,lr)
              print('learning rate re-reduced..')
              self.flag_3=True
            elif(self.flag_3==True):
              self.early_stopping(epochinst)
              return True,optimizer
    return False,optimizer
  def early_stopping(self,epochinst):
    print('training is terminating so as to prevent further overfitting')
    epochinst.set_total(epochinst.get_step())

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
