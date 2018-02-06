from scapy.all import *
import numpy as np
import utils



def readPcap(filename):
    packets = np.zeros(1460)
    count = 0
    data = rdpcap(filename)
    sessions = data.sessions()
    for id, session in sessions.items():
        for packet in session:
            if IP in packet:
                ip_layer = packet[IP]
                time = packet.time
                dst = ip_layer.dst
                src = ip_layer.src
                transport_layer = ip_layer.payload
                protocol = transport_layer.name
                dport = transport_layer.dport
                sport = transport_layer.sport
                payload = transport_layer.payload

readPcap('DRTV_akamai_small.pcap')