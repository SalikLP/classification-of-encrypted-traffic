import datetime

import utils

if __name__ == '__main__':
    loaddir = "D:/Data/h5/"
    headers = [1, 2, 4, 8, 16]
    for headersize in headers:
        savedir= "D:/Data/h5/" + str(headersize) + "/"
        now = datetime.datetime.now()
        savename = "extracted_%d-%.2d%.2d_%.2d%.2d" % (headersize, now.day, now.month, now.hour, now.minute)
        utils.saveextractedheaders(loaddir, savedir, savename, num_headers=headersize)