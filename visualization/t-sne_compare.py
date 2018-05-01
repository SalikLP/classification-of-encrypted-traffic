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
seed = 0
num_headers = 16
dirs = ["C:/Users/salik/Documents/Data/LinuxChrome/{}/".format(num_headers),
        "C:/Users/salik/Documents/Data/WindowsChrome/{}/".format(num_headers)]
# dirs = ["E:/Data/h5/https/", "E:/Data/h5/netflix/"]
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
num_rows = data.shape[0]
columns = data.columns
print(columns)

# step 2: get features (x) and  convert it to numpy array
standard_scaler = StandardScaler()
x_stds = []
ys = []
class_labels = []
for data in dataframes:
    x = da.getbytes(data, num_headers*54)
# x = da.getbytes(data, 1460)
    x_std = standard_scaler.fit_transform(x)
    x_stds.append(x_std)
# step 4: get class labels y and then encode it into number
# get class label data
    y = data['label'].values
    # encode the class label
    class_labels = np.unique(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    ys.append(y)
class_labels1 = ["Linux " + x for x in class_labels]
class_labels2 = ["Windows " + x for x in class_labels]
# step 5: split the data into training set and test set
test_percentage = 0.1
x_tests = []
y_tests = []
for i, x_std in enumerate(x_stds):
    x_train, x_test, y_train, y_test = train_test_split(x_std, ys[i], test_size=test_percentage, random_state=seed)
    x_tests.append(x_test)
    y_tests.append(y_test)

x_test = np.append(x_tests[0], x_tests[1], axis=0)
y_test = np.append(y_tests[0], y_tests[1], axis=0)
first_set_length = len(y_tests[0])
print(first_set_length)

# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
plot_savename = "t-sne_16headers_windows_linux_perplexity"
from sklearn.manifold import TSNE
perplexities = [30.0]

from matplotlib import rcParams
# Make room for xlabel which is otherwise cut off
rcParams.update({'figure.autolayout': True})

for perplexity in perplexities:
    print("Starting perplexity: {}".format(perplexity))
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=seed, verbose=2)
    x_test_2d = tsne.fit_transform(x_test)
    x_test_2d_0 = x_test_2d[:first_set_length, :]
    x_test_2d_1 = x_test_2d[first_set_length:, :]

    # scatter plot the sample points among 7 classes
    # markers = ('s', 'd', 'o', '^', 'v', ".", ",", "<", ">", "8", "p", "P", "*", "h", "H", "+", "x", "X", "D", "|", "_")
    color_map = {0: '#487fff', 1: '#2ee3ff', 2: '#4eff4e', 3: '#ffca43', 4: '#ff365e', 5: '#d342ff', 6:'#626663'}
    plt.figure()
    for idx, cl in enumerate(np.unique(y_test)):
        # Plot first dataset as +
        plt.scatter(x=x_test_2d_0[y_tests[0] == cl, 0], y=x_test_2d_0[y_tests[0] == cl, 1],
                    marker="+", s=30,
                    c=color_map[idx],
                    label=class_labels1[cl])
        # Plot second dataset as o with no fill
        plt.scatter(x=x_test_2d_1[y_tests[1] == cl, 0], y=x_test_2d_1[y_tests[1] == cl, 1],
                    marker="o", facecolors="None", s=30, linewidths=1,
                    edgecolors=color_map[idx],
                    label=class_labels2[cl])
    plt.xlabel('X in t-SNE')
    plt.ylabel('Y in t-SNE')
    plt.legend(loc='lower right')
    plt.title('t-SNE visualization with perplexity: {}'.format(perplexity))
    plt.tight_layout()
#     plt.savefig('{0}{1}.png'.format(plot_savename, int(perplexity)), dpi=300)
plt.show()