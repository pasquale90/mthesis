#define experiment's mode

#define experiment mode
class mode_class:
  def __init__(self, mode_atr):
    if (mode_atr == 44100):
      self.mode='44100'
    elif (mode_atr == 22050):
      self.mode = '22050'
    else:
      print(f'{mode_atr} input attribute is not valid.Please insert 44100 or 22050')
  def get_mode(self):
    return self.mode




#make a class attempt
#attempt = 1


#mode
#flags: data(path),main(define_model), model(params.mode)
