import numpy as np
import pandas as pd
import utils
# import seaborn as sns
import glob
import os
from scipy import stats, integrate
import matplotlib.pyplot as plt
# sns.set(color_codes=True)


def remove_checksum(nrheaders):
    # nrheaders = 1
    read_dir = '/home/mclrn/Data/linux/'
    headers = '{0}/'.format(nrheaders)
    dataframes = []
    for fullname in glob.iglob(read_dir + headers + '*.h5'):
        filename = os.path.basename(fullname)
        df = utils.load_h5(read_dir + headers, filename)
        dataframes.append(df)
    # create one large dataframe
    df = pd.concat(dataframes)
    # df = pd.read_hdf('/home/mclrn/Data/salik_windows/{0}/'.format(nrheaders), key="extracted_{0}".format(nrheaders))
    df = df.sample(frac=1).reset_index(drop=True)
    values, counts = np.unique(df['label'], return_counts=True)
    print(values, counts)

    #selector = df['label'] == 'youtube'

    values = df['bytes'].values
    bytes = np.zeros((values.shape[0], nrheaders * 54))
    for i, v in enumerate(values):
        payload = np.zeros(nrheaders * 54, dtype=np.uint8)
        payload[:v.shape[0]] = v
        bytes[i] = payload

    #mean = np.mean(bytes, axis=0)
    #min = np.min(bytes, axis=0)
    #max = np.max(bytes, axis=0)
    #print(np.max(bytes[0:, 23]))  # Protocol field if value = 6 then TCP if value = 17 the UDP
    bytes_no_checksum = []
    for j, b in enumerate(bytes):
        if b[23] == 6:
            # TCP
            # if bytenumber in (50, 51):
            #   return "Checksum (TCP header)"
            for i in range(nrheaders):
                b[i * 54 + 50] = 0
                b[i * 54 + 51] = 0
                b[i * 54 + 24] = 0
                b[i * 54 + 25] = 0
        elif b[23] == 17:
            # UDP
            # if bytenumber in (40,41)
            # return "UDP Checksum (UDP Header)"
            for i in range(nrheaders):
                b[i * 42 + 40] = 0
                b[i * 42 + 41] = 0
                b[i * 42 + 24] = 0
                b[i * 42 + 25] = 0
        else:
            print("Byte was not 6 nor 17 but: %d" % bytes[23])

        bytes_no_checksum.append(b)
    new_data = {'bytes': bytes_no_checksum, 'label': df['label'].values}
    new_df = pd.DataFrame(new_data)
    # print(df)
    # print(new_df)
    save_dir = read_dir + "no_checksum/" + headers
    # if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    new_df.to_hdf(save_dir +
                  "extracted_{0}-no_checksum".format(nrheaders) + '.h5',
                  key='extracted_{0}'.format(nrheaders), mode='w')


nr = [1,2,4,8,16]
for n in nr:
    remove_checksum(n)


#sns.distplot(bytes[0:,370], kde=False, rug=True)

'''
for index, m in enumerate(mean):
    if m > 0 and min[index] != max[index]:
        print(index, min[index], max[index], mean[index])
        sns.distplot(bytes[0:,index], kde=False, rug=True)
        plt.show()
'''
#plt.show()
#plt.savefig("dist.png")

#for i in mean:
#    print("%.2f" % i)


