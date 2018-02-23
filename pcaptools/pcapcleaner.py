import os
import glob
dir = '../../Data/'

for fullname in glob.iglob(dir + '*.pcap'):
    dir, filename = os.path.split(fullname)
    command = 'tshark -r %s -2 -R "!(eth.dst[0]&1) && !(tcp.port==5901) && ip" -w %s/filtered/%s' % (fullname,dir, filename)
    os.system(command)
