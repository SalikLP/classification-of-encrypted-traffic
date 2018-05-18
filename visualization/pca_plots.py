import pca.dataanalyzer as da, pca.pca as pca
import glob
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import utils


seed = 0
num_headers = 16
dirs =  ["E:/Data/LinuxChrome/{}/".format(num_headers),
         "E:/Data/WindowsSalik/{}/".format(num_headers),
         "E:/Data/WindowsAndreas/{}/".format(num_headers),
         "E:/Data/WindowsFirefox/{}/".format(num_headers),
         "E:/Data/WindowsChrome/{}/".format(num_headers),
                      ]
# dirs = ["C:/Users/salik/Documents/Data/h5/https/", "C:/Users/salik/Documents/Data/h5/netflix/"]
# dirs = ["C:/Users/salik/Documents/Data/WindowsAndreas/{}/".format(num_headers)]
# step 1: get the data
dataframes = []
num_examples = 0
for dir in dirs:
    for fullname in glob.iglob(dir + '*.h5'):
        filename = os.path.basename(fullname)
        df = utils.load_h5(dir, filename)
        dataframes.append(df)
        num_examples = len(df.values)
    # create one large dataframe
data = pd.concat(dataframes)
data = data.sample(frac=0.1, random_state=seed).reset_index(drop=True)
num_rows = data.shape[0]
columns = data.columns
print(columns)

# step 3: get features (x) and scale the features
# get x and convert it to numpy array
x = da.getbytes(data, num_headers*54)
standard_scaler = StandardScaler()
# x_stds = []
# ys = []
# for data in dataframes:
# x = da.getbytes(data, 1460)

# x = [[-0.5, -0.5],
#      [-0.5, 0.5],
#      [0.5, -0.5],
#      [0.5, 0.5],
#      [-0.4, -0.4],
#      [-0.4, 0.4],
#      [0.4, -0.4],
#      [0.4, 0.4]
#      ]
x_std = standard_scaler.fit_transform(x)
y = data['label'].values
# y = ['drtv', 'netflix', 'youtube', 'twitch', 'twitch', 'youtube', 'netflix', 'drtv']
# encode the class label
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


p = pca.runpca(x_std, num_comp=25)
z = pca.componentprojection(x_std, p)
pca.plotprojection(z, 0, y, class_labels)
pca.plotvarianceexp(p, 25)
pca.showplots()
# plot_savename = "PCA_header_all_cumulative"
# plt.savefig('{0}.png'.format(plot_savename), dpi=300)