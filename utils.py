import multiprocessing

from scapy.all import *
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import pandas as pd
import glob


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
    frametimes = []
    dsts = []
    srcs = []
    protocols = []
    dports = []
    sports = []
    bytes = []
    labels = []

    print("Total packages: %d" % totalPackets)
    time_r = time.clock()
    time_read = time_r - time_s
    print("Time to read PCAP: " + str(time_read))
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
            if (count % (percentage * 5) == 0):
                print(str(count / percentage) + '%')
            count += 1
    time_t = time.clock()
    print("Time spend: %ds" % (time_t - time_r))
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
    df = pd.read_hdf(dir + "/" + filename, key=filename.split('-')[0])
    timeE = time.clock()
    loadTime = timeE - timeS
    print("Time to load " + filename + ": " + str(loadTime))
    return df


def plotHex(hexvalues, filename):
    '''
        Plot an example as an image
        hexvalues: list of byte values
        average: allows for providing more than one list of hexvalues and create an average over all
    '''

    size = 39
    hex_placeholder = [0] * (size * size)  # create placeholder of correct size

    if (type(hexvalues[0]) is np.ndarray):
        print("Multiple payloads")
        for hex_list in hexvalues:
            hex_placeholder[0:len(hex_list)] += hex_list  # overwrite zero values with values of
        hex_placeholder = np.array(hex_placeholder) / len(hexvalues)  # average the elements of the placeholder
    else:
        print("Single payload")
        hex_placeholder[0:len(hexvalues)] = hexvalues  # overwrite zero values with values of

    canvas = np.reshape(np.array(hex_placeholder), (size, size))
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(canvas, cmap='gray')
    plt.title(filename)
    plt.show()
    return canvas


def pad_string_elements_with_zero(payloads):
    # Assume max payload to be 1460 bytes but as each byte is now 2 hex digits we take double length
    max_payload_len = 1460 * 2
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


def packetanonymizer(packet):
    """"
    Takes a packet as a bytestring in hex format and convert to unsigned 8bit integers [0-255]
    Sets the header fields which contain MAC, IP and Port information to 0
    """
    # Should work with TCP and UDP

    p = np.fromstring(packet, dtype=np.uint8)
    # set MACs to 0
    p[0:12] = 0
    # Remove IP checksum
    p[24:26] = 0
    # set IPs to 0
    p[26:34] = 0
    # set ports to 0
    p[34:36] = 0
    p[36:38] = 0

    # IP protocol field check if TCP
    if p[23] == 6:
        #Remove TCP checksum
        p[50:52] = 0
    else:
        # Remove UDP checksum
        p[40:42] = 0
    return p


def extractdatapoints(dataframe, filename, num_headers=15, session=True):
    """"
    Extracts the concatenated header datapoints from a dataframe while anonomizing the individual header
    :returns a dataframe with datapoints (bytes) and labels
    """
    group_by = dataframe.sort_values(['time']).groupby(['ip.dst', 'ip.src', 'port.dst', 'port.src'])
    gb_dict = dict(list(group_by))
    data_points = []
    labels = []
    filenames = []
    sessions = []
    done = set()
    num_too_short = 0
    # Iterate over sessions
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
        # extract num_headers of packets (one row each)
        packets = v['bytes'].values[:num_headers]
        headers = []
        label = v['label'].iloc[0]
        protocol = v['protocol'].iloc[0]
        # Skip session if UDP and not youtube
        if protocol == 'TCP' or label == 'youtube':
            continue
        # For each packet
        packetindex = 0
        headeradded = 0

        while headeradded < num_headers:
            if packetindex < len(packets):
                p = packets[packetindex]
                packetindex += 1
            else:
                break
            p_an = packetanonymizer(p)

            # assuming a session utilize the same protocol throughout
            # Extract headers (TCP = 54 Bytes, UDP = 42 Bytes - Maybe + 4 Bytes for VLAN tagging) from x first packets of session/flow
            header = np.zeros(54, dtype=np.uint8)
            if protocol == 'TCP':
                # TCP
                header[:54] = p_an[:54]
            else:
                # UDP
                header[:42] = p_an[:42]  # pad zeros

            # Skip if header packet is fragmented
            if (0 < header[20] < 64) or header[21] != 0:
                continue
            headers.append(header)
            headeradded += 1

        # Concatenate headers as the feature vector
        if len(headers) == num_headers:
            feature_vector = np.concatenate(headers).ravel()
            data_points.append(feature_vector)
            labels.append(label)
            filenames.append(filename)
            sessions.append(k)
    d = {'filename': filenames, 'session': sessions, 'bytes': data_points, 'label': labels}
    return pd.DataFrame(data=d)


def saveheaderstask(filelist, num_headers, session, dataframes):
    datapointslist = []
    for fullname in filelist:
        load_dir, filename = os.path.split(fullname)
        print("Loading: {0}".format(filename))
        df = load_h5(load_dir, filename)
        datapoints = extractdatapoints(df, filename, num_headers, session)
        datapointslist.append(datapoints)

    # Extend the shared dataframe
    dataframes.extend(datapointslist)


def saveextractedheaders(load_dir, save_dir, savename, num_headers=15, session=True):
    """"
    Extracts datapoints from all .h5 files in train_dir and saves the them in a new .h5 file
    :param load_dir: The directory to load from
    :param save_dir: The directory to save the extracted headers
    :param savename: The filename to save
    :param num_headers: The amount of headers to use as datapoint
    :param session: session or flow
    """
    manager = multiprocessing.Manager()
    dataframes = manager.list()
    filelist = glob.glob(load_dir + '*.h5')
    filesplits = split_list(filelist, 4)

    threads = []
    for split in filesplits:
        # create a thread for each
        t = multiprocessing.Process(target=saveheaderstask, args=(split, num_headers, session, dataframes))
        threads.append(t)
        t.start()
    # create one large dataframe

    for t in threads:
        t.join()
        print("Process joined: ", t)
    data = pd.concat(dataframes)
    key = savename.split('-')[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data.to_hdf(save_dir + savename + '.h5', key=key, mode='w')


# saveextractedheaders('./', 'extracted-0103_1136')
# read_pcap('../Data/', 'drtv-2302_1031')
def split_list(list, chunks):
    '''
    Takes a list an splits it to equal sized chunks.
    :param list: list to split
    :param chunks: number of chunks (int)
    :return: a list containing chunks (lists) as elements
    '''
    avg = len(list) / float(chunks)
    out = []
    last = 0.0

    while last < len(list):
        out.append(list[int(last):int(last + avg)])
        last += avg

    return out


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap(name='Blues'), save=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    from matplotlib import rcParams
    # Make room for xlabel which is otherwise cut off
    rcParams.update({'figure.autolayout': True})

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Accuracy: {0}".format(title.split("acc")[1]))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='vertical')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save:
        i = 0
        filename = "{}".format(title)
        while os.path.exists('{}{:d}.png'.format(filename, i)):
            i += 1
        plt.savefig('{}{:d}.png'.format(filename, i), dpi=300)
    else:
        plt.draw()
    plt.gcf().clear()



def plot_metric_graph(x_list, y_list,x_label="Datapoints", y_label="Accuracy",
                          title='Metric list', save=False):
    from matplotlib import rcParams
    # Make room for xlabel which is otherwise cut off
    rcParams.update({'figure.autolayout': True})

    plt.plot(x_list, y_list)
    # Calculate min and max of y scale
    ymin = np.min(y_list)
    ymin = np.floor(ymin * 10) / 10
    ymax = np.max(y_list)
    ymax = np.ceil(ymax * 10) / 10
    plt.ylim(ymin, ymax)
    plt.title("{0}".format(title))
    plt.tight_layout()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if save:

        i = 0
        filename = "{}".format(title)
        while os.path.exists('{}{:d}.png'.format(filename, i)):
            i += 1
        plt.savefig('{}{:d}.png'.format(filename, i), dpi=300)
    plt.draw()
