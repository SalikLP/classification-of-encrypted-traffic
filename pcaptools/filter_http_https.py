import utils
import trafficgen.config as cf
import socket

Conf = cf.Conf
all_ips = []
for http in Conf.http_urls:
    url = http.split('/')[2]
    ip_list = []
    ais = socket.getaddrinfo(url, 0, 0, 0, 0)
    for result in ais:
        ip_list.append(result[-1][0])
    ip_list = list(set(ip_list))
    all_ips.append(ip_list)

flat_list = [ip for ips in all_ips for ip in ips]

dir = '../'
filename = 'http-https-blandet_1902_1300'
dataframe = utils.filter_pcap_by_ip(dir,filename, flat_list, 'http')
utils.save_dataframe_h5(dataframe,dir, 'http-browse-1902_1410')

