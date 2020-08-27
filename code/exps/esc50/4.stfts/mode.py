#define experiment's mode

#define experiment mode
class mode_class:
  def __init__(self, mode_atr):
    if (mode_atr == 'exp4'):
      self.mode='exp4'
    elif (mode_atr == 'exp8'):
      self.mode = 'exp8'
    else:
      print(f'{mode_atr} input attribute is not valid.Please insert <exp4> or <exp8>')
  def get_mode(self):
    return self.mode

