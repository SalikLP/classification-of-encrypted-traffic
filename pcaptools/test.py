import utils
import pandas as pd
import tf_utils
import tensorflow as tf
import dataset
import numpy

dir = '../../Data/'
# filename = 'netflix-0602_1042'
# # filename = 'DRTV-akamai_small'
#
# utils.save_pcap(dir, filename)
# print("Save done!")
dataset.read_data_sets(dir, one_hot=True)
# df = utils.load_h5(dir + filename+'.h5', key=filename.split('-')[0])
# print("Load done!")
# print(df.shape)
# payloads = df['payload'].values
# payloads = utils.pad_elements_with_zero(payloads)
# df['payload'] = payloads
#
# # Converting hex string to list of int... Maybe takes to long?
# payloads = [[int(i, 16) for i in list(x)] for x in payloads]
# np_payloads = numpy.array(payloads)
# # dataset = DataSet(np_payloads, df['label'].values)
# x, y = dataset.next_batch(10)
# batch_size = 100
# features = {'payload': payloads}
#
#
# gb = df.groupby(['ip.dst', 'ip.src', 'port.dst', 'port.src'])
#
#
# l = dict(list(gb))
#
#
# s = [[k, len(v)] for k, v in sorted(l.items(), key=lambda x: len(x[1]), reverse=True)]


print("DONE")
