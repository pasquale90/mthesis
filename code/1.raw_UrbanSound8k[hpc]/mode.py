#define experiment mode
class mode_class:
  def __init__(self, mode_atr):
    if (mode_atr == 8):
      self.mode='8'
    elif (mode_atr == 16):
      self.mode='16'
    elif (mode_atr == 22):
      self.mode = '22'
    elif (mode_atr == 32):
      self.mode = '32'
    else:
      print(f'{mode_atr} input attribute is not valid.Please insert 8 or 16 or 22 or 32')
  def get_mode(self):
    return self.mode
