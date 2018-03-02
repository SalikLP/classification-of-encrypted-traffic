import utils
import glob
import os
import pandas as pd
import numpy as np


def getmeanstd(dataframe, label):
    labels = dataframe['label'] == label
    values = dataframe[labels]['bytes'].values
    bytes = np.zeros((values.shape[0], values[0].shape[0]))
    for i, v in enumerate(values):
        bytes[i] = v

    # Ys = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    mean = np.mean(bytes, axis=0)
    mean_sub = np.subtract(bytes, mean)
    std = mean_sub / np.std(bytes, axis=0)
    return mean, mean_sub, std

train_dir = '../../Data/h5/extracted/'
dataframes = []
for fullname in glob.iglob(train_dir + '*.h5'):
    filename = os.path.basename(fullname)
    df = utils.load_h5(train_dir, filename)
    dataframes.append(df)
# create one large dataframe
data = pd.concat(dataframes)
dr_mean, dr_mean_sub, dr_std = getmeanstd(data, 'drtv')
nf_mean, nf_mean_sub, nf_std = getmeanstd(data, 'netflix')
print(dr_mean-nf_mean)


