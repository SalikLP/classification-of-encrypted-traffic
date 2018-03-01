import utils
import glob
import os
import pandas as pd
import numpy as np


train_dir = '../../Data/h5/extracted/'
dataframes = []
for fullname in glob.iglob(train_dir + '*.h5'):
    filename = os.path.basename(fullname)
    df = utils.load_h5(train_dir, filename)
    dataframes.append(df)
# create one large dataframe
data = pd.concat(dataframes)
drtv = data['label'] == 'drtv'
values = data[drtv]['bytes'].values
values = np.reshape(values, (values.shape[0], 810))
zero = np.zeros((values.shape[0], 810))
new_values = np.add(values, zero)
# Ys = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# values = values.astype(np.uint32)
X_m = np.mean(new_values, axis=0)
X_m= np.reshape(X_m, (1, X_m.shape[0]))
X_ms= np.subtract(new_values, X_m)
# X_ms = (new_values[0] - X_m)
X_std = X_ms/np.std(new_values, axis=0)
print(np.linalg.matrix_rank(X_std))
print(values)
