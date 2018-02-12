import utils

dir = '../../Data/'
filenames = ['http-download']
for filename in filenames:
    utils.save_pcap(dir, filename)
    print("Save done!", filename)