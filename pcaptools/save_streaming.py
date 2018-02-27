import sys
# Add the utils folder path to the sys.path list
sys.path.append('/home/mclrn/dlproject/')
import utils
import glob
import os
from threading import Thread
import multiprocessing

def split_list(list, chunks):
    avg = len(list) / float(chunks)
    out = []
    last = 0.0

    while last < len(list):
        out.append(list[int(last):int(last + avg)])
        last += avg

    return out



def save_streaming_task(files):
    for full_name in files:
        dir_n, filename = os.path.split(full_name)
        print('Currently saving file: ', filename)
        utils.save_pcap(dir, filename, session_threshold)



if __name__ == "__main__":
    dir = '../Data/'
    session_threshold = 10000

    # Load all files
    files = []
    for fullname in glob.iglob(dir + '*.pcap'):
        files.append(fullname)

    splits = 4
    files_splits = split_list(files,splits)

    for file_split in files_splits:
        # create a thread for each
        t1 = multiprocessing.Process(target=save_streaming_task, args=(file_split,))
        t1.start()





