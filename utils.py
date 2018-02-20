from scapy.all import *
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd


def save_pcap(dir, filename):
    df = read_pcap(dir, filename)
    save_dataframe_h5(df, dir, filename)


def save_dataframe_h5(df, dir, filename):
    key = filename.split('-')[0]
    df.to_hdf(dir + filename + '.h5', key=key, mode='w')


def read_pcap(dir, filename):
    time_s = time.clock()
    count = 0
    label = filename.split('-')[0]
    print("Read PCAP, label is %s" % label)
    data = rdpcap(dir + filename + '.pcap')
    totalPackets = len(data)
    percentage = int(totalPackets / 100)
    # Workaround/speedup for pandas append to dataframe
    frametimes =[]
    dsts = []
    srcs = []
    protocols = []
    dports = []
    sports = []
    bytes = []
    labels = []

    print("Total packages: %d" % totalPackets)
    time_r = time.clock()
    time_read = time_r-time_s
    print("Time to read PCAP: "+ str(time_read))
    sessions = data.sessions()
    for id, session in sessions.items():
        for packet in session:
            if IP in packet and (UDP in packet or TCP in packet):
                ip_layer = packet[IP]
                transport_layer = ip_layer.payload
                frametimes.append(packet.time)
                dsts.append(ip_layer.dst)
                srcs.append(ip_layer.src)
                protocols.append(transport_layer.name)
                dports.append(transport_layer.dport)
                sports.append(transport_layer.sport)
                # Save the raw byte string
                raw_payload = raw(packet)
                bytes.append(raw_payload)
                labels.append(label)
                if(count%(percentage*5) == 0):
                    print(str(count/percentage) + '%')
                count += 1
    time_t = time.clock()
    print("Time spend: %ds" % (time_t-time_r))
    d = {'time': frametimes,
         'ip.dst': dsts,
         'ip.src': srcs,
         'protocol': protocols,
         'port.dst': dports,
         'port.src': sports,
         'bytes': bytes,
         'label': labels}
    df = pd.DataFrame(data=d)
    time_e = time.clock()
    total_time = time_e - time_s
    print("Time to convert PCAP to dataframe: " + str(total_time))
    return df


def filter_pcap_by_ip(dir, filename, ip_list, label):
    time_s = time.clock()
    count = 0
    print("Read PCAP, label is %s" % label)
    data = rdpcap(dir + filename + '.pcap')
    totalPackets = len(data)
    percentage = int(totalPackets / 100)
    # Workaround/speedup for pandas append to dataframe
    frametimes =[]
    dsts = []
    srcs = []
    protocols = []
    dports = []
    sports = []
    bytes = []
    labels = []

    print("Total packages: %d" % totalPackets)
    time_r = time.clock()
    time_read = time_r-time_s
    print("Time to read PCAP: "+ str(time_read))
    for packet in data:
        if IP in packet and \
                (UDP in packet or TCP in packet) and \
                (packet[IP].dst in ip_list or packet[IP].src in ip_list):
            ip_layer = packet[IP]
            transport_layer = ip_layer.payload
            frametimes.append(packet.time)
            dsts.append(ip_layer.dst)
            srcs.append(ip_layer.src)
            protocols.append(transport_layer.name)
            dports.append(transport_layer.dport)
            sports.append(transport_layer.sport)
            # Save the raw byte string
            raw_payload = raw(packet)
            bytes.append(raw_payload)
            labels.append(label)
            if(count%(percentage*5) == 0):
                print(str(count/percentage) + '%')
            count += 1
    time_t = time.clock()
    print("Time spend: %ds" % (time_t-time_r))
    d = {'time': frametimes,
         'ip.dst': dsts,
         'ip.src': srcs,
         'protocol': protocols,
         'port.dst': dports,
         'port.src': sports,
         'bytes': bytes,
         'label': labels}
    df = pd.DataFrame(data=d)
    time_e = time.clock()
    total_time = time_e - time_s
    print("Time to convert PCAP to dataframe: " + str(total_time))
    return df


def load_h5(dir, filename):
    timeS = time.clock()
    df = pd.read_hdf(dir + filename, key=filename.split('-')[0])
    timeE = time.clock()
    loadTime = timeE-timeS
    print("Time to load " + filename + ": " + str(loadTime))
    return df

def plotHex(hexvalues, filename):
    '''
        Plot an example as an image
        hexvalues: list of byte values
        average: allows for providing more than one list of hexvalues and create an average over all
    '''

    size = 39
    hex_placeholder = [0]*(size*size) #create placeholder of correct size


    if(type(hexvalues[0]) is np.ndarray):
      print("Multiple payloads")
      for hex_list in hexvalues:
        hex_placeholder[0:len(hex_list)] += hex_list  # overwrite zero values with values of
      hex_placeholder = np.array(hex_placeholder)/len(hexvalues) # average the elements of the placeholder
    else:
      print("Single payload")
      hex_placeholder[0:len(hexvalues)] = hexvalues #overwrite zero values with values of

    canvas = np.reshape(np.array(hex_placeholder),(size,size))
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(canvas, cmap='gray')
    plt.title(filename)
    plt.show()
    return canvas


def pad_elements_with_zero(payloads):
    # Assume max payload to be 1460 bytes but as each byte is now 2 hex digits we take double length
    max_payload_len = 1460*2
    # Pad with '0'
    payloads = [s.ljust(max_payload_len, '0') for s in payloads]
    return payloads


def hash_elements(payloads):
    return payloads

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")


def packetAnonymizer(packet):
    # Should work with TCP and UDP
    p = np.fromstring(packet, dtype=np.uint8)
    # set MACs to 0
    p[0:12] = 0
    # set IPs to 0
    p[26:34] = 0
    # set ports to 0
    p[34:36] = 0
    p[36:38] = 0
    return p
