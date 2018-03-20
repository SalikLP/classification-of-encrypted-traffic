import pcap.pcaptools as pcap



if __name__ == '__main__':
    pcap.process_pcap_to_h5('/home/mclrn/Data/', '/home/mclrn/Data/h5/', session_threshold=5000)