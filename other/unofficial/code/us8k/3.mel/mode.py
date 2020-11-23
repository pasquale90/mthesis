#define experiment's mode

#define experiment mode
class mode_class:
  def __init__(self, mode_atr):
    if (mode_atr == 80):
      self.mode='80'
    elif (mode_atr == 128):
      self.mode = '128'
    elif (mode_atr == 360):
      self.mode='360'
    else:
      print(f'{mode_atr} input attribute is not valid.Please insert 80 or 128')
  def get_mode(self):
    return self.mode




#make a class attempt
#attempt = 1


#mode
#flags: data(path),main(define_model), model(params.mode)
