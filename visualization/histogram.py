import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import utils
import glob, os
import pca.dataanalyzer as da, pca.pca as pca
from sklearn.metrics import accuracy_score

# visulaize the important characteristics of the dataset
import matplotlib.pyplot as plt
data_len = 1460
seed = 0
# dirs = ["C:/Users/salik/Documents/Data/LinuxChrome/{}/".format(num_headers),
#         "C:/Users/salik/Documents/Data/WindowsFirefox/{}/".format(num_headers),
#         "C:/Users/salik/Documents/Data/WindowsChrome/{}/".format(num_headers),
#         "C:/Users/salik/Documents/Data/WindowsSalik/{}/".format(num_headers),
#         "C:/Users/salik/Documents/Data/WindowsAndreas/{}/".format(num_headers)]
dirs = ["E:/Data/h5/https/", "E:/Data/h5/netflix/"]

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
data.sample(frac=1, random_state=seed).reset_index(drop=True)
num_rows = data.shape[0]
columns = data.columns
print(columns)

# step 2: get features (x) and convert it to numpy array
x = da.getbytes(data, data_len)

# step 3: get class labels y and then encode it into number
# get class label data
y = data['label'].values

# encode the class label
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# step 4: split the data into training set and test set
test_percentage = 0.5
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_percentage, random_state=seed)

plot_savename = "histogram_payload"

from matplotlib import rcParams
# Make room for xlabel which is otherwise cut off
rcParams.update({'figure.autolayout': True})



# scatter plot the sample points among 5 classes
# markers = ('s', 'd', 'o', '^', 'v', ".", ",", "<", ">", "8", "p", "P", "*", "h", "H", "+", "x", "X", "D", "|", "_")
color_map = {0: '#487fff', 1: '#d342ff', 2: '#4eff4e', 3: '#2ee3ff', 4: '#ffca43', 5:'#ff365e', 6:'#626663'}
plt.figure()
for idx, cl in enumerate(np.unique(y_test)):
    # Get count of unique values
    values, counts = np.unique(x_test[y_test == cl], return_counts=True)
    n, bins, patches = plt.hist(values, weights=counts, bins=256, facecolor=color_map[idx+1], label=class_labels[cl],  alpha=0.8)
    plt.legend(loc='upper right')
    plt.title('Histogram of : {}'.format(class_labels))
    plt.tight_layout()
# plt.savefig('{0}{1}.png'.format(plot_savename, int(perplexity)), dpi=300)
plt.show()