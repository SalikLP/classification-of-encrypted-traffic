import utils
import glob
import os
import pandas as pd
import numpy as np
import math
import pca as p


def getbytes(dataframe, payload_length=810):
    values = dataframe['bytes'].values
    bytes = np.zeros((values.shape[0], payload_length))
    for i, v in enumerate(values):
        payload = np.zeros(payload_length, dtype=np.uint8)
        payload[:v.shape[0]] = v
        bytes[i] = payload
    return bytes


def getmeanstd(dataframe, label):
    labels = dataframe['label'] == label
    bytes = getbytes(dataframe[labels])
    # values = dataframe[labels]['bytes'].values
    # bytes = np.zeros((values.shape[0], values[0].shape[0]))
    # for i, v in enumerate(values):
    #     bytes[i] = v

    # Ys = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    mean = np.mean(bytes, axis=0)
    mean_sub = np.subtract(bytes, mean)
    std = mean_sub / np.std(bytes, axis=0)
    return mean, mean_sub, std


def byteindextoheaderfield(number, TCP=True):
    if TCP:
        bytenumber = number % 54
    else:
        bytenumber = number % 42

    if bytenumber in range(6):
        return "Destination MAC"
    if bytenumber in range(6, 12):
        return "Source MAC"
    if bytenumber in (12, 13):
        return "Eth. Type"
    if bytenumber == 14:
        return "IP Version and header length"
    if bytenumber == 15:
        return "Explicit Congestion Notification"
    if bytenumber in (16, 17):
        return "Total Length (IP header)"
    if bytenumber in (18, 19):
        return "Identification (IP header)"
    if bytenumber in (20, 21):
        return "Fragment offset (IP header)"
    if bytenumber == 22:
        return "Time to live (IP header)"
    if bytenumber == 23:
        return "Protocol (IP header)"
    if bytenumber in (24, 25):
        return "Header checksum (IP header)"
    if bytenumber in range(26, 30):
        return "Source IP (IP header)"
    if bytenumber in range(30, 34):
        return "Destination IP (IP header)"
    if bytenumber in (34, 35):
        return "Source Port (TCP/UDP header)"
    if bytenumber in (36, 37):
        return "Destination Port (TCP/UDP header)"
    if bytenumber in range(38, 42):
        if TCP:
            return "Sequence number (TCP header)"
        elif bytenumber in (38, 39):
            return "Length of data (UDP Header)"
        else:
            return "UDP Checksum (UDP Header)"
    if bytenumber in range(42, 46):
        return "ACK number (TCP header)"
    if bytenumber == 46:
        return "TCP Header length or Nonce (TCP header)"
    if bytenumber == 47:
        return "TCP FLAGS (CWR, ECN-ECHO, ACK, PUSH, RST, SYN, FIN) (TCP header)"
    if bytenumber in (48, 49):
        return "Window size (TCP header)"
    if bytenumber in (50, 51):
        return "Checksum (TCP header)"
    if bytenumber in (52, 53):
        return "Urgent Pointer (TCP header)"


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
mean_diff = dr_mean - nf_mean
sort_diff = (-abs(mean_diff)).argsort() #Sort on absolute values in decending order
for i in range(10):
    packetnumber = math.ceil(sort_diff[i] / 54)
    bytenumber = sort_diff[i] % 54
    print('Index %i is bytenumber %i in packet: %i' % (sort_diff[i],bytenumber, packetnumber), byteindextoheaderfield(sort_diff[i]))

bytes = getbytes(data)
labels = data['label']
pca = p.runpca(bytes, 50)
p.plotvarianceexp(pca, 50)
Z = p.componentprojection(bytes, pca)
p.plotprojection(Z, 0, labels)
p.showplots()
# id_max = np.argmax(mean_diff)
# print(byteindextoheaderfield(id_max))
# id_min = np.argmin(mean_diff)
# print(dr_mean-nf_mean)




