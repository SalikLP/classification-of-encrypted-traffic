import datetime
import utils
import glob
import os
import numpy as np
import pandas as pd


if __name__ == '__main__':
    loaddir = "E:/Data/h5/"
    labels = ['https', 'netflix']
    max_packet_length = 1514
    for label in labels:
        print("Starting label: " + label)
        savedir = loaddir + label + "/"
        now = datetime.datetime.now()
        savename = "payload_%s-%.2d%.2d_%.2d%.2d" % (label, now.day, now.month, now.hour, now.minute)
        filelist = glob.glob(loaddir + label + '*.h5')
        # Try only one of each file
        fullname = filelist[0]
        # for fullname in filelist:
        load_dir, filename = os.path.split(fullname)
        print("Loading: {0}".format(filename))
        df = utils.load_h5(load_dir, filename)
        packets = df['bytes'].values
        payloads = []
        labels = []
        filenames = []
        for packet in packets:
            if len(packet) == max_packet_length:
                # Extract the payload from the packet should have length 1460
                payload = packet[54:]
                p = np.fromstring(payload, dtype=np.uint8)
                payloads.append(p)
                labels.append(label)
                filenames.append(filename)
        d = {'filename': filenames, 'bytes': payloads, 'label': labels}
        dataframe = pd.DataFrame(data=d)
        key = savename.split('-')[0]
        dataframe.to_hdf(savedir + savename + '.h5', key=key, mode='w')
        # utils.saveextractedheaders(loaddir, savedir, savename, num_headers=headersize)
        print("Done with label: " + label)
