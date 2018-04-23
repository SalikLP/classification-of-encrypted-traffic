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
num_headers = 8
dirs = ["C:/Users/salik/Documents/Data/WindowsAndreas/{}/".format(num_headers)]
# step 1: get the data
dataframes = []
for dir in dirs:
    for fullname in glob.iglob(dir + '*.h5'):
        filename = os.path.basename(fullname)
        df = utils.load_h5(dir, filename)
        dataframes.append(df)
    # create one large dataframe
data = pd.concat(dataframes)
data.sample(frac=1, random_state=seed).reset_index(drop=True)
num_rows = data.shape[0]
columns = data.columns
print(columns)

# step 3: get features (x) and scale the features
# get x and convert it to numpy array
x = da.getbytes(data, num_headers*54)
# x = da.getbytes(x, num_headers*54)
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)
# step 4: get class labels y and then encode it into number
# get class label data
y = data['label'].values
# encode the class label
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# step 5: split the data into training set and test set
test_percentage = 0.99
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=test_percentage, random_state=seed)
#
# p = pca.runpca(x_test, num_comp=2)
# z = pca.componentprojection(x_test, p)
# pca.plotprojection(z, 0, y_test, class_labels)
# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
from sklearn.manifold import TSNE
perplexities = [50.0]
for perplexity in perplexities:
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=5000, random_state=seed)
    x_test_2d = tsne.fit_transform(x_test)

    # scatter plot the sample points among 5 classes
    # markers = ('s', 'd', 'o', '^', 'v', ".", ",", "<", ">", "8", "p", "P", "*", "h", "H", "+", "x", "X", "D", "|", "_")
    color_map = {0: 'orangered', 1: 'royalblue', 2: 'lightgreen', 3: 'darkorchid', 4: 'teal', 5: 'darkslategrey', 6:'darkgreen', 7:'darkgrey'}
    plt.figure()
    for idx, cl in enumerate(np.unique(y_test)):
        plt.scatter(x=x_test_2d[y_test == cl, 0], y=x_test_2d[y_test == cl, 1], c=color_map[idx], label=class_labels[cl])
    plt.xlabel('X in t-SNE')
    plt.ylabel('Y in t-SNE')
    plt.legend(loc='upper left')
    plt.title('t-SNE visualization with perplexity: {}'.format(perplexity))
plt.show()