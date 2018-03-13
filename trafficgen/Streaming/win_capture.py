import os
import datetime


def captureTraffic(interfaceNumber, duration, dir, name):
    '''
    Interfacenumber: specifies the network interface that should be captured (use tshark -D to list the options)
    duration: specifies the number of seconds that the capture should go on
    dir: is the folder to which the pcap file of the capture should be saved
    name: is the name of that pcap file (this will always be appended by date and time)
    '''
    now = datetime.datetime.now()
    file = dir + "/%s-%.2d%.2d_%2d%.2d.pcap" % (name, now.day, now.month, now.hour, now.minute)

    # makedir if it does not exist
    if not os.path.isdir(dir):
        os.mkdir(dir)

    open(file, "w") # overwrites if file already exists
    os.system("tshark -i %d -a duration:%d -w %s port not 5901 and ip and not broadcast and not multicast" % (interfaceNumber, duration, file))



'''
To test
captureTraffic(1,10, 'C:/users/arhjo/desktop', "test")
'''