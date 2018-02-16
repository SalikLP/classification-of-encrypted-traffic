import utils
from scapy.all import *

dir = '../../Data/'
filenames = ['DRTV-akamai_small2']
for filename in filenames:
    data = rdpcap(dir + filename + '.pcap')
    sessions = data.sessions()
    for id, session in sessions.items():
        for packet in session:
            anon = utils.packetAnonymizer(raw(packet))