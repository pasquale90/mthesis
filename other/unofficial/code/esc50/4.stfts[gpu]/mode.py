#define experiment's mode

class mode_class:
  def __init__(self, mode_atr):
    if (mode_atr == 1):
      self.mode='1'
    elif (mode_atr == 2):
      self.mode = '2'
    else:
      print(f'{mode_atr} input attribute is not valid.Please insert <1> or <2>')
  def get_mode(self):
    return self.mode
