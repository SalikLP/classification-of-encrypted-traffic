import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.datasets import base
import glob
import os
import utils
import pandas as pd
from sklearn.preprocessing import LabelEncoder


_label_encoder = LabelEncoder()
class DataSet(object):

    def __init__(self,
                   payloads,
                   labels,
                   one_hot=False,
                   dtype=dtypes.float32,
                   seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
          raise TypeError('Invalid payload dtype %r, expected uint8 or float32' %
                          dtype)

        assert payloads.shape[0] == labels.shape[0], (
              'payloads.shape: %s labels.shape: %s' % (payloads.shape, labels.shape))
        self._num_examples = payloads.shape[0]

        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            payloads = payloads.astype(np.float32)
            payloads = np.multiply(payloads, 1.0 / 255.0)

        self._payloads = payloads
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def payloads(self):
        return self._payloads

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._payloads = self.payloads[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            payloads_rest_part = self._payloads[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._payloads = self.payloads[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._payloads[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((payloads_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._payloads[start:end], self._labels[start:end]


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.int8)
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(dataframe, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
    dataframe: A pandas dataframe object.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

    Returns:
    labels: a 1D uint8 numpy array.

    Raises:
    ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting labels', )
    labels = dataframe['label'].values
    labels = _label_encoder.fit_transform(labels)
    if one_hot:
        return dense_to_one_hot(labels, num_classes)
    return labels


def read_data_sets(train_dir,
                   one_hot=False,
                   dtype=dtypes.float32,
                   validation_size=1000,
                   test_size=1000,
                   seed=None):

    dataframes = []
    for fullname in glob.iglob(train_dir+'*.h5'):
        filename =os.path.basename(fullname)
        df = utils.load_h5(train_dir + filename, key=filename.split('-')[0])
        dataframes.append(df)
    # create one large dataframe
    data = pd.concat(dataframes)
    # shuffle the dataframe and reset the index
    data = data.sample(frac=1).reset_index(drop=True)
    num_classes = len(data['label'].unique())
    labels = extract_labels(data, one_hot=one_hot, num_classes=num_classes)

    payloads = data['payload'].values
    # array_pl = list(raw_payload) this converts raw bytestring to list of int

    # Converting raw bytestring to np array and pad with zero up to 1460 length
    tmp_payloads = []
    for x in payloads:
        payload = np.zeros(1460, dtype=np.uint8)
        pl = np.fromstring(x, dtype=np.uint8)
        payload[:pl.shape[0]] = pl
        tmp_payloads.append(payload)

    # payloads = [np.fromstring(x) for x in payloads]
    payloads = np.array(tmp_payloads)

    if not 0 <= validation_size <= len(payloads):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(payloads), validation_size))

    test_payloads = payloads[:test_size]
    test_labels = labels[:test_size]
    val_payloads = payloads[test_size:(validation_size+test_size)]
    val_labels = labels[test_size:(validation_size+test_size)]
    train_payloads = payloads[(validation_size+test_size):]
    train_labels = labels[(validation_size+test_size):]
    options = dict(dtype=dtype, seed=seed)

    train = DataSet(train_payloads, train_labels, **options)
    validation = DataSet(val_payloads, val_labels, **options)
    test = DataSet(test_payloads, test_labels, **options)

    return base.Datasets(train=train, validation=validation, test=test)

