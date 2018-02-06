from scapy.all import *
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd


def save_pcap(filename):
    df = read_pcap(filename)
    save_dataframe_h5(df, filename)


def save_dataframe_h5(df, filename):
    key = filename.split('.')[0]
    df.to_hdf(filename + '.h5', key=key)


def read_pcap(filename):
    timeS = time.clock()
    df = pd.DataFrame(columns=['time', 'ip.dst', 'ip.src', 'protocol', 'port.dst', 'port.src', 'payload', 'label'])
    count = 0
    label = filename.split('-')[0]
    data = rdpcap(filename)
    timeR = time.clock()
    timeRead = timeR-timeS
    print("Time to read PCAP: "+ str(timeRead))
    sessions = data.sessions()
    for id, session in sessions.items():
        for packet in session:
            if IP in packet:
                ip_layer = packet[IP]
                frametime = packet.time
                dst = ip_layer.dst
                src = ip_layer.src
                transport_layer = ip_layer.payload
                protocol = transport_layer.name
                dport = transport_layer.dport
                sport = transport_layer.sport
                payload = transport_layer.payload.original
                df.loc[count] = [frametime, dst, src, protocol, dport, sport, payload, label]
                count += 1
    timeE = time.clock()
    totalTime = timeE - timeS
    print("Time to convert PCAP to dataframe: " + str(totalTime))
    return df


def load_h5(filename, key=''):
    timeS = time.clock()
    df = pd.read_hdf(filename, key=key)
    timeE = time.clock()
    loadTime = timeE-timeS
    print("Time to load " + filename + ": " + str(loadTime))
    return df

def plotHex(hexvalues, filename):
    '''
        Plot an example as an image
    '''
    size = 39
    canvas = np.zeros((size,size))
    padding = (size*size)-len(hexvalues)
    #odd = if padding % 2 == 0 false  else true
    odd = False
    if padding %2 == 0:
        odd = False
    else:
        odd = True
    padding = int(np.floor(padding/2))
    hexvalues = np.pad(hexvalues,padding,'constant')
    print(hexvalues)
    if odd:
        hexvalues = np.append(hexvalues,[0])
    canvas = np.reshape(np.array(hexvalues),(size,size))
    plt.figure(figsize=(4,4))
    plt.axis('off')
    plt.imshow(canvas, cmap='gray')
    plt.title(filename)
    plt.show()
