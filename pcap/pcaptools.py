import pandas as pd
from scapy.all import *
import glob
import os
import multiprocessing

from utils import split_list


def pcap_cleaner(dir):
    """
    This method can be used for cleaning pcap files.
    :param dir: Directory containing the pcap files that should be filtered
    :return: None
    """
    for fullname in glob.iglob(dir + '*.pcap'):
        dir, filename = os.path.split(fullname)
        command = 'tshark -r %s -2 -R "!(eth.dst[0]&1) && !(tcp.port==5901) && ip" -w %s/filtered/%s' % (fullname,dir, filename)
        os.system(command)


def save_pcap_task(files, save_dir, session_threshold):
    """
    This method takes all files in a list (full path names) and uses the method save_pcap that converts a pcap to h5 format
    :param files:
    :param session_threshold:
    :return:
    """
    for fullname in files:
        print('Currently saving file: ', fullname)

        save_pcap(fullname, save_dir, session_threshold)


def process_pcap_to_h5(read_dir, save_dir, session_threshold=5000):
    """
    Use this method to process all pcap files in a directory to a h5 format.
    Session threshold is used to filter out all sessions containing fewer packets
    :param save_dir:
    :param read_dir: Directory containing pcap files that should be converted into h5 format
    :param session_threshold: Threshold to filter out session with less packets
    :return: None
    """
    h5files = []

    for h5 in glob.iglob(save_dir + '*.h5'):
        h5files.append(os.path.basename(h5))
    # Load all files
    files = []
    for fullname in glob.iglob(read_dir + '*.pcap'):
        filename = os.path.basename(fullname)
        h5name = filename +'.h5'
        if h5name in h5files:
            os.rename(fullname, read_dir + '/processed_pcap/' + filename)
        else:
            files.append(fullname)

    splits = 3
    files_splits = split_list(files, splits)

    for file_split in files_splits:
        # create a thread for each
        t1 = multiprocessing.Process(target=save_pcap_task, args=(file_split, save_dir, session_threshold))
        t1.start()


def save_pcap(fullname, save_dir, session_threshold=0):
    """
    This method read a pcap file and saves it to an h5 dataframe.
    The file is overwritten if it already exists.
    :param dir: The folder containing the pcap file
    :param filename: The name of the pcap file
    :return: Nothing
    """
    dir_n, filename = os.path.split(fullname)
    df = read_pcap(fullname, filename, session_threshold)
    key = filename.split('-')[0]
    df.to_hdf(save_dir + filename + '.h5', key=key, mode='w')


def read_pcap(fullname, filename, session_threshold=0):
    """
    This method will extract the packets of the major session within the pcap file. It will label the packets according
    to the filename.
    The method excludes packets between local/internal ip adresses (ip.src and ip.dst startswith 10.....)
    The method finds the major sessions by counting the packets for each session and calculate a threshold dependent
    on the session with most packets. All sessions with more packets than the threshold value is extracted and placed
    in the dataframe.

    :param dir: The directory in which the pcap file is located. Should end with a /
    :param filename: The name of the pcap file. It is expected to contain the label of the data before the first - char
    :return: A dataframe containing the extracted packets.
    """
    time_s = time.clock()
    label = filename.split('-')[0]
    print("Read PCAP, label is %s" % label)
    if not filename.endswith('.pcap'):
        filename += '.pcap'
    data = rdpcap(fullname)
    # Workaround/speedup for pandas append to dataframe
    frametimes =[]
    dsts = []
    srcs = []
    protocols = []
    dports = []
    sports = []
    bytes = []
    labels = []

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
    """
    Custom session extractor to use for scapy to group bi directional sessions instead of a uni directional flows.
    :param p: packet as used by scapy
    :return: session string to use a key in dict
    """
    sess = "Other"
    if 'Ether' in p:
        if 'IP' in p:
            src = p[IP].src
            dst = p[IP].dst
            if NTP in p:
                if src.startswith('10.') or src.startswith('192.168.'):
                    sess = p.sprintf("NTP %IP.src%:%r,UDP.sport% > %IP.dst%:%r,UDP.dport%")
                elif dst.startswith('10.') or dst.startswith('192.168.'):
                    sess = p.sprintf("NTP %IP.dst%:%r,UDP.dport% > %IP.src%:%r,UDP.sport%")
            elif 'TCP' in p:
                if src.startswith('10.') or src.startswith('192.168.'):
                    sess = p.sprintf("TCP %IP.src%:%r,TCP.sport% > %IP.dst%:%r,TCP.dport%")
                elif dst.startswith('10.') or dst.startswith('192.168.'):
                    sess = p.sprintf("TCP %IP.dst%:%r,TCP.dport% > %IP.src%:%r,TCP.sport%")
            elif 'UDP' in p:
                if src.startswith('10.') or src.startswith('192.168.'):
                    sess = p.sprintf("UDP %IP.src%:%r,UDP.sport% > %IP.dst%:%r,UDP.dport%")
                elif dst.startswith('10.') or dst.startswith('192.168.'):
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