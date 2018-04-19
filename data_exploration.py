import pca.dataanalyzer
import utils
import glob
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pca.dataanalyzer import byteindextoheaderfield
'''
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

'''
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

def createBoxplotsFromColumns(title,param, param1):

    plt.title(title)
    plt.boxplot([param,param1])

    plt.savefig('boxplots/boxplot:%s.png' % title,dpi=300)
    plt.gcf().clear()

def compare_data(path1, path2, nrheaders):
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for fullname in glob.iglob(path1 + '*.h5'):
        filename = os.path.basename(fullname)
        df1 = utils.load_h5(path1, filename)

    for fullname in glob.iglob(path2 + '*.h5'):
        filename = os.path.basename(fullname)
        df2 = utils.load_h5(path2, filename)

    classes = []
    # find all classes
    for label in set(df1['label']):
        classes.append(label)
    print(set(df1['label']))
    print(set(df2['label']))
    # filter on classes
    for c in classes:
        ## Exclude youtube as it contains both UDP and TCP
        if(c == 'youtube'):
            continue


        # create selector
        df1_selector = df1['label'] == c
        df2_selector = df2['label'] == c


        df1_values = df1[df1_selector]['bytes'].values
        df2_values = df2[df2_selector]['bytes'].values

        df1_bytes = np.zeros((df1_values.shape[0], nrheaders * 54))
        df2_bytes = np.zeros((df2_values.shape[0], nrheaders * 54))

        for i, v in enumerate(df1_values):
            payload = np.zeros(nrheaders * 54, dtype=np.uint8)
            payload[:v.shape[0]] = v
            df1_bytes[i] = payload

        for i, v in enumerate(df2_values):
            payload = np.zeros(nrheaders * 54, dtype=np.uint8)
            payload[:v.shape[0]] = v
            df2_bytes[i] = payload



        # Extract byte 23 to determine the protocol.
        TCP = True if int(df2_bytes[0][23]) == 6 else False


        df1_mean = np.mean(df1_bytes, axis=0)
        df2_mean = np.mean(df2_bytes, axis=0)

        df1_min = np.min(df1_bytes, axis=0)
        df2_min = np.min(df2_bytes, axis=0)

        df1_max = np.max(df1_bytes, axis=0)
        df2_max = np.max(df2_bytes, axis=0)



        for index, mean in enumerate(df1_mean):
            if(index % 25 == 0):
                print(c,index)
            if df1_mean[index] > 0 or df2_mean[index] > 0:
                if(int(df1_min[index]) != int(df1_max[index]) or int(df2_min[index]) != int(df2_max[index])):
                    if(int(df1_mean[index]) != int(df2_mean[index]) or int(df1_min[index]) != int(df2_min[index]) or int(df1_max[index]) != int(df2_max[index])):
                        print(index, " : ", int(df1_mean[index]), ' : ' , int(df2_mean[index]), int(df1_min[index]), int(df2_min[index]), int(df1_max[index]), int(df2_max[index]))
                        headername = byteindextoheaderfield(index, TCP)
                        headername = headername.replace('/',' ')
                        createBoxplotsFromColumns(c+headername +':'+str(index), df1_bytes[:,index], df2_bytes[:,index])


compare_data('/home/mclrn/Data/linux/no_checksum/8/', '/home/mclrn/Data/windows_firefox/no_checksum/8/',8)

