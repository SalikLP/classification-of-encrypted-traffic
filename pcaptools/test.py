import utils
import pandas as pd

dir = '../../Data/'
# filename = 'http-download'
filename = 'DRTV-akamai_small'

# utils.save_pcap(dir, filename)
# print("Save done!")
df = utils.load_h5(dir + filename+'.h5', key=filename.split('-')[0])
print("Load done!")
print(df.shape)
payloads = df['payload'].values
payloads = utils.pad_elements_with_zero(payloads)
df['payload'] = payloads


gb = df.groupby(['ip.dst', 'ip.src', 'port.dst', 'port.src'])


l = dict(list(gb))


s = [[k, len(v)] for k, v in sorted(l.items(), key=lambda x: len(x[1]), reverse=True)]
print("DONE")
