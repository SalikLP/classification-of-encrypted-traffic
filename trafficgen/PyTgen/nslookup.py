import socket
import config

Conf = config.Conf
# ip_list = []
# ais = socket.getaddrinfo('en.wikipedia.org', 0, 0, 0, 0)
# for result in ais:
#   ip_list.append(result[-1][0])
# ip_list = list(set(ip_list))
# print(ip_list)

count = 0
for http in Conf.https_urls:
    url = http.split('/')[2]
    ip_list = []
    ais = socket.getaddrinfo(url, 0, 0, 0, 0)
    for result in ais:
      ip_list.append(result[-1][0])
    ip_list = list(set(ip_list))
    count +=1
    print(http, ip_list, count)