import utils

filename = 'https.pcap'

utils.save_pcap(filename)
print("Save done!")
df = utils.load_h5(filename+'.h5', key=filename.split('.')[0])
print("Load done!")
print(df.shape)
