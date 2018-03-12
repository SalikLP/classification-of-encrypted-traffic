from scapy.all import *
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import glob


def save_pcap(dir, filename, session_threshold=None):
    '''
    This method read a pcap file and saves it to an h5 dataframe.
    The file is overwritten if it already exists.
    :param dir: The folder containing the pcap file
    :param filename: The name of the pcap file
    :return: Nothing
    '''
    df = read_pcap(dir, filename, session_threshold)
    key = filename.split('-')[0]
    df.to_hdf(dir + 'h5/' + filename + '.h5', key=key, mode='w')


def read_pcap(dir, filename, session_threshold=None):
    '''
    This method will extract the packets of the major session within the pcap file. It will label the packets according
    to the filename.
    The method excludes packets between local/internal ip adresses (ip.src and ip.dst startswith 10.....)
    The method finds the major sessions by counting the packets for each session and calculate a threshold dependent
    on the session with most packets. All sessions with more packets than the threshold value is extracted and placed
    in the dataframe.

    :param dir: The directory in which the pcap file is located. Should end with a /
    :param filename: The name of the pcap file. It is expected to contain the label of the data before the first - char
    :return: A dataframe containing the extracted packets.
    '''
    time_s = time.clock()
    count = 0
    label = filename.split('-')[0]
    print("Read PCAP, label is %s" % label)
    if not filename.endswith('.pcap'):
        filename += '.pcap'
    data = rdpcap(dir + filename)
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
    sessions = data.sessions(session_extractor=session_extractor)
    for id, session in sessions.items():
        if session_threshold is not None:
            if len(session) < session_threshold:
                continue
        for packet in session:
            # Check that the packet is transferred by either UDP or TCP and ensure that it is not a packet between to local/internal IP adresses (occurs when using vnc and such)
            if IP in packet and (UDP in packet or TCP in packet) and not (packet[IP].dst.startswith('10.') and packet[IP].src.startswith('10.')):
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
    '''
    This method can be used  to extract certain packets (associted with specified ip adresses) from a pcap file.
    The method expects user knowledge of the communication protocol used by the specified ip adresses for the label to be correct.

    The label is intended to be either http or https.
    :param dir: The directory in which the pcap file is located
    :param filename: The name of the pcap file that should be loaded
    :param ip_list: A list of ip adresses of interest. Packets with either ip.src or ip.dst in ip_list will be extracted
    :param label: The label that should be applied to the extracted packets. Note that the ip_list should contain
                  adresses that we know communicate over http or https in order to match with the label
    :return: A dataframe containing information about the extracted packets from the pcap file.
    '''
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


def session_extractor(p):
    sess = "Other"
    if 'Ether' in p:
        if 'IP' in p:
            src = p[IP].src
            dst = p[IP].dst
            if NTP in p:
                if src.startswith('10.'):
                    sess = p.sprintf("NTP %IP.src%:%r,UDP.sport% > %IP.dst%:%r,UDP.dport%")
                elif dst.startswith('10.'):
                    sess = p.sprintf("NTP %IP.dst%:%r,UDP.dport% > %IP.src%:%r,UDP.sport%")
            elif 'TCP' in p:
                if src.startswith('10.'):
                    sess = p.sprintf("TCP %IP.src%:%r,TCP.sport% > %IP.dst%:%r,TCP.dport%")
                elif dst.startswith('10.'):
                    sess = p.sprintf("TCP %IP.dst%:%r,TCP.dport% > %IP.src%:%r,TCP.sport%")
            elif 'UDP' in p:
                if src.startswith('10.'):
                    sess = p.sprintf("UDP %IP.src%:%r,UDP.sport% > %IP.dst%:%r,UDP.dport%")
                elif dst.startswith('10.'):
                    sess = p.sprintf("UDP %IP.dst%:%r,UDP.dport% > %IP.src%:%r,UDP.sport%")
            elif 'ICMP' in p:
                sess = p.sprintf("ICMP %IP.src% > %IP.dst% type=%r,ICMP.type% code=%r,ICMP.code% id=%ICMP.id%")
            else:
                sess = p.sprintf("IP %IP.src% > %IP.dst% proto=%IP.proto%")
        elif 'ARP' in p:
            sess = p.sprintf("ARP %ARP.psrc% > %ARP.pdst%")
        else:
            sess = p.sprintf("Ethernet type=%04xr,Ether.type%")
    return sess


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


def pad_string_elements_with_zero(payloads):
    # Assume max payload to be 1460 bytes but as each byte is now 2 hex digits we take double length
    max_payload_len = 1460*2
    # Pad with '0'
    payloads = [s.ljust(max_payload_len, '0') for s in payloads]
    return payloads


def pad_arrays_with_zero(payloads, payload_length=810):
    tmp_payloads = []
    for x in payloads:
        payload = np.zeros(payload_length, dtype=np.uint8)
        # pl = np.fromstring(x, dtype=np.uint8)
        payload[:x.shape[0]] = x
        tmp_payloads.append(payload)

    # payloads = [np.fromstring(x) for x in payloads]
    return np.array(tmp_payloads)

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
    """"
    Takes a packet as a bytestring in hex format and convert to unsigned 8bit integers [0-255]
    Sets the header fields which contain MAC, IP and Port information to 0
    """
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


def extractdatapoints(dataframe, num_headers=15, session=True):
    """"
    Extracts the concatenated header datapoints from a dataframe while anonomizing the individual header
    :returns a dataframe with datapoints (bytes) and labels
    """
    group_by = dataframe.sort_values(['time']).groupby(['ip.dst', 'ip.src', 'port.dst', 'port.src'])
    gb_dict = dict(list(group_by))
    data_points = []
    labels = []
    done = set()
    num_too_short = 0
    for k, v in gb_dict.items():
        # v is a DataFrame
        # k is a tuple (src, dst, sport, dport)
        if k in done:
            continue
        done.add(k)
        if session:
            other_direction_key = (k[1], k[0], k[3], k[2])
            other_direction = gb_dict[other_direction_key]
            v = pd.concat([v, other_direction]).sort_values(['time'])
            done.add(other_direction_key)
        if len(v) < num_headers:
            num_too_short += 1
            continue
        packets = v['bytes'].values[:num_headers]
        headers = []
        for i in range(num_headers):
            p = packets[i]
            p_an = packetAnonymizer(p)
            protocol = v['protocol'].iloc[0]
            # Extract headers (TCP = 54 Bytes, UDP = 42 Bytes - Maybe + 4 Bytes for VLAN tagging) from x first packets of session/flow
            if protocol == 'TCP':
                # TCP
                header = p_an[:54]
            else:
                # UDP
                header = p_an[:42]
            headers.append(header)
        # Concatenate headers as the feature vector
        feature_vector = np.concatenate(headers).ravel()
        data_points.append(feature_vector)
        labels.append(v['label'].iloc[0])
    d = {'bytes': data_points, 'label': labels}
    return pd.DataFrame(data=d)


def saveextractedheaders(train_dir, savename):
    """"
    Extracts datapoints from all .h5 files in train_dir and saves the them in a new .h5 file
    """
    dataframes = []
    for fullname in glob.iglob(train_dir + '*.h5'):
        filename = os.path.basename(fullname)
        df = load_h5(train_dir, filename)
        datapoints = extractdatapoints(df)
        dataframes.append(datapoints)
    # create one large dataframe
    data = pd.concat(dataframes)
    key = savename.split('-')[0]
    data.to_hdf(train_dir + 'extracted/' + savename + '.h5', key=key, mode='w')


# saveextractedheaders('./', 'extracted-0103_1136')
# read_pcap('../Data/', 'drtv-2302_1031')