import numpy as np


class EarlyStopping:
    """Stop training when a monitored quantity has stopped improving.
  Arguments:
      min_delta: minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: number of epochs with no improvement
          after which training will be stopped.
  """

    def __init__(self,
                 min_delta=0,
                 patience=0):
        self.stop_training = False
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        self.monitor_op = np.less
        self.min_delta *= -1
        self.best = np.Inf

    def on_train_begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        self.stop_training = False

    def on_epoch_end(self, epoch, current):
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
            self.wait += 1

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print('Epoch {}: early stopping'.format(self.stopped_epoch))
