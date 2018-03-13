import os


def captureTraffic(interfaceNumber, duration, dir, file):
    '''
    Interfacenumber: specifies the network interface that should be captured (use tshark -D to list the options)
    duration: specifies the number of seconds that the capture should go on
    dir: is the folder to which the pcap file of the capture should be saved
    name: is the name of that pcap file (this will always be appended by date and time)
    '''
    # makedir if it does not exist
    if not os.path.isdir(dir):
        os.mkdir(dir)
    #open(file, "w") # overwrites if file already exists
    os.system("echo %s |sudo -S tshark -i %d -a duration:%d -w %s port not 5901 and ip and not broadcast and not multicast" % ('Napatech10',interfaceNumber, duration, file))
    os.system("echo %s |sudo -S chown mclrn:mclrn %s" % ('Napatech10', file))



'''
To test
captureTraffic(1,10, 'C:/users/arhjo/desktop', "test")
'''


def cleanup(file):
    os.system("echo %s |sudo -S rm %s" % ('Napatech10', file))
