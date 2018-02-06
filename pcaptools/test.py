import utils

dir = '../../Data/'
filename = 'netflix-0602_1042'


utils.save_pcap(dir,filename)
print("Save done!")
df = utils.load_h5(dir + filename+'.h5', key=filename.split('-')[0])
print("Load done!")
print(df.shape)
