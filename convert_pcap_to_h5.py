import pcap.pcaptools as pcap



if __name__ == '__main__':
    read_dir = "D:/Data/"
    save_dir = "D:/Data/h5/"
    pcap.process_pcap_to_h5(read_dir, save_dir, session_threshold=5000)