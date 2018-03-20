import pca.dataanalyzer
import utils
import glob
import os
import pandas as pd

train_dir = '../../Data/h5/extracted/'
dataframes = []
for fullname in glob.iglob(train_dir + '*.h5'):
    filename = os.path.basename(fullname)
    df = utils.load_h5(train_dir, filename)
    dataframes.append(df)
# create one large dataframe
data = pd.concat(dataframes)
print(len(data[data['label'] == 'drtv']))
print(len(data[data['label'] == 'http']))
print(len(data[data['label'] == 'https']))
print(len(data[data['label'] == 'netflix']))
print(len(data[data['label'] == 'youtube']))


# dr_mean, dr_mean_sub, dr_std = getmeanstd(data, 'drtv')
# nf_mean, nf_mean_sub, nf_std = getmeanstd(data, 'netflix')
# mean_diff = dr_mean - nf_mean
# sort_diff = (-abs(mean_diff)).argsort() #Sort on absolute values in decending order
# for i in range(10):
#     packetnumber = math.ceil(sort_diff[i] / 54)
#     bytenumber = sort_diff[i] % 54
#     print('Index %i is bytenumber %i in packet: %i' % (sort_diff[i],bytenumber, packetnumber), byteindextoheaderfield(sort_diff[i]))
#
# bytes = getbytes(data)
# labels = data['label']
# pca = p.runpca(bytes, 50)
# # p.plotvarianceexp(pca, 50)
# Z = p.componentprojection(bytes, pca)
# for pc in range(17):
#     p.plotprojection(Z, pc, labels)
# p.showplots()
# id_max = np.argmax(mean_diff)
# print(byteindextoheaderfield(id_max))
# id_min = np.argmin(mean_diff)
# print(dr_mean-nf_mean)
