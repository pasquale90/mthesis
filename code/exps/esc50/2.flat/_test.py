import numpy as np
import pandas as pd
import paths


genres_filename = "3_80_esc50_overalF1_1.csv"
prev = pd.read_csv(paths.results_path+genres_filename, index_col=False)

#prev.columns[0].name='ep'
prev.index.name = 'ep'
#prev.reset_index(level=prev.columns[0])
data = {'mean_train_loss' : 1, 'mean_valid_loss' : 2,
	 'micro_f1' :3, 'macro_f1' :4 }
#prev.loc['epoch 30'] = [ 1.0, 2.0, 3.0, 4.0]
#prev.append(['epoch 30', 1,2,3,4])
print(prev)
prev.to_csv(paths.results_path+'test.csv')
