import utils

dir = '../../Data/'
filename = 'DRTV-akamai_small'
df = utils.load_h5(dir + filename+'.h5', key=filename.split('-')[0])
print("Load done!")
print(df.shape)
df1 = df.groupby(['ip.dst', 'ip.src', 'port.dst', 'port.src'])
d = dict(list(df1))

print(d)