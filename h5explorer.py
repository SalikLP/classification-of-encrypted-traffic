import numpy as np
import glob
import os
import pandas as pd
# read_folder
# pick class
# read h5 belonging to class
# find min, max of byte on index (column)

# return name of h5 with min, max bytevalue
import utils


def examineH5files(read_dir, label, nrheaders, byteindex):
    for fullname in glob.iglob(read_dir + '*.h5'):
        filename = os.path.basename(fullname)
        if(filename.startswith(label)):
            df1 = utils.load_h5(read_dir, filename)
            df = utils.extractdatapoints(df1,nrheaders)
        else:
            continue

examineH5files('/home/mclrn/Data/h5/', 'twitch', 8, byteindex=144)