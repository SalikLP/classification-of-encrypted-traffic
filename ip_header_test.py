import glob
import os
import utils
import numpy as np
import multiprocessing

num_headers = 1

load_dir = 'E:/Data/h5/'


def ipheadertask(filelist):
    j = 1
    for fullname in filelist:
        print("Loading filenr: {}".format(j))
        load_dir, filename = os.path.split(fullname)
        df = utils.load_h5(load_dir, filename)
        frames = df['bytes'].values
        for i, frame in enumerate(frames):
            p = np.fromstring(frame, dtype=np.uint8)
            if p[14] != 69:
                print("IP Header length not 20! in file {0}".format(filename))
        j += 1

if __name__ == '__main__':
    filelist = glob.glob(load_dir + '*.h5')
    filesplits = utils.split_list(filelist, 4)

    threads = []
    for split in filesplits:
        # create a thread for each
        t = multiprocessing.Process(target=ipheadertask, args=(split,))
        threads.append(t)
        t.start()
    # create one large dataframe

    for t in threads:
        t.join()
        print("Process joined: ", t)
