import utils
import trafficgen.PyTgen.config as cf
import socket


def save_dataframe_h5(df, dir, filename):
    key = filename.split('-')[0]
    df.to_hdf(dir + filename + '.h5', key=key)
    
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

http_list = [ip for ips in all_ips for ip in ips]

all_ips = []
for http in Conf.https_urls:
    url = http.split('/')[2]
    ip_list = []
    ais = socket.getaddrinfo(url, 0, 0, 0, 0)
    for result in ais:
        ip_list.append(result[-1][0])
    ip_list = list(set(ip_list))
    all_ips.append(ip_list)
https_list = [ip for ips in all_ips for ip in ips]

dir = 'C:/Users/admin/Desktop/'
filename = 'http_https_04-04-2018'
http_dataframe = utils.filter_pcap_by_ip(dir,filename, http_list, 'http')
save_dataframe_h5(http_dataframe,dir, 'http-browse-0404_2300')

https_dataframe = utils.filter_pcap_by_ip(dir,filename, https_list, 'https')
save_dataframe_h5(https_dataframe, dir, 'https-browse-0404_2330')


