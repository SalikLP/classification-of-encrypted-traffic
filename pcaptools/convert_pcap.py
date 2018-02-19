import utils

dir = '../../Data/'
filenames = ['http-download', 'netflix-0602_1042filtered']
for filename in filenames:
    utils.save_pcap(dir, filename)
    print("Save done!", filename)