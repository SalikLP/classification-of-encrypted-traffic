import tensorflow as tf


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def ffn_layer(name, inputs, hidden_units, activation=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    input_dim = inputs.get_shape().as_list()[1]
    # use xavier glorot intitializer as regular uniform dist did not work
    weight_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
    with tf.variable_scope(name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = tf.get_variable('W', [input_dim, hidden_units], initializer=weight_initializer)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = tf.get_variable('b', [hidden_units], initializer=tf.zeros_initializer)
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(inputs, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = activation(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations
        # layer = tf.layers.dense(inputs=inputs, units=hidden_units, activation=activation)
        # return layer


def conv_layer_1d(name, inputs, num_filters=1, filter_size=(1,1), strides=1, activation=None):
    """"A simple function to create a 1D conv layer"""
    with tf.variable_scope(name):
        # TensorFlow operation for convolution
        layer = tf.layers.conv1d(inputs=inputs,
                                 filters=num_filters,
                                 kernel_size=filter_size,
                                 strides=strides,
                                 padding='same',
                                 activation=activation)
        return layer


def conv_layer_2d(name, inputs, num_filters=1, filter_size=(1,1), strides=(1,1), activation=None):
    """"A simple function to create a 2D conv layer"""
    with tf.variable_scope(name):
        # TensorFlow operation for convolution
        layer = tf.layers.conv2d(inputs=inputs,
                                 filters=num_filters,
                                 kernel_size=filter_size,
                                 strides=strides,
                                 padding='same',
                                 activation=activation)
        return layer


def max_pool_layer(inputs, name, pool_size=(1, 1), strides=(1, 1), padding='same'):
    """"A simple function to create a max pool layer"""
    with tf.variable_scope(name):
        # TensorFlow operation for max pooling
        layer = tf.layers.max_pooling2d(inputs=inputs,
                                        pool_size=pool_size,
                                        strides=strides,
                                        padding=padding)
        return layer


def dropout(inputs, keep_prob=1.0):
    with tf.name_scope('dropout'):
        # keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        return tf.nn.dropout(inputs, keep_prob)


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

#
# mnist_data = input_data.read_data_sets('MNIST_data',
#                                        one_hot=True,   # Convert the labels into one hot encoding
#                                        dtype='float32', # rescale images to `[0, 1]`
#                                        reshape=False, # Don't flatten the images to vectors
#                                        )

# # Simple MNIST test of the layers
#
# tf.reset_default_graph()
# num_classes = 10
# cm = conf.ConfusionMatrix(num_classes)
# height, width, nchannels = 28, 28, 1
# gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
# filters_1 = 16
# kernel_size_1 = (5,5)
# pool_size_1 = (2,2)
# x_pl = tf.placeholder(tf.float32, [None, height, width, nchannels], name='xPlaceholder')
# y_pl = tf.placeholder(tf.float64, [None, num_classes], name='yPlaceholder')
# y_pl = tf.cast(y_pl, tf.float32)
#
# x = conv_layer_2d('layer1', x_pl, filters_1, kernel_size_1, activation=tf.nn.relu)
# x = max_pool_layer(x, 'max_pool', pool_size = pool_size_1, strides=pool_size_1)
# x = tf.contrib.layers.flatten(x)
#
# y = ffn_layer('output_layer', x, hidden_units=num_classes, activation=tf.nn.softmax)
#
# with tf.variable_scope('loss'):
#     # computing cross entropy per sample
#     cross_entropy = -tf.reduce_sum(y_pl * tf.log(y + 1e-8), reduction_indices=[1])
#
#     # averaging over samples
#     cross_entropy = tf.reduce_mean(cross_entropy)
#
# with tf.variable_scope('training'):
#     # defining our optimizer
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
#
#     # applying the gradients
#     train_op = optimizer.minimize(cross_entropy)
#
# with tf.variable_scope('performance'):
#     # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
#     correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))
#
#     # averaging the one-hot encoded vector
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#
# # Training Loop
# batch_size = 100
# max_epochs = 10
#
# valid_loss, valid_accuracy = [], []
# train_loss, train_accuracy = [], []
# test_loss, test_accuracy = [], []
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print('Begin training loop')
#
#     try:
#         while mnist_data.train.epochs_completed < max_epochs:
#             _train_loss, _train_accuracy = [], []
#
#             ## Run train op
#             x_batch, y_batch = mnist_data.train.next_batch(batch_size)
#             fetches_train = [train_op, cross_entropy, accuracy]
#             feed_dict_train = {x_pl: x_batch, y_pl: y_batch}
#             _, _loss, _acc = sess.run(fetches_train, feed_dict_train)
#
#             _train_loss.append(_loss)
#             _train_accuracy.append(_acc)
#
#             ## Compute validation loss and accuracy
#             if mnist_data.train.epochs_completed % 1 == 0 \
#                     and mnist_data.train._index_in_epoch <= batch_size:
#                 train_loss.append(np.mean(_train_loss))
#                 train_accuracy.append(np.mean(_train_accuracy))
#
#                 fetches_valid = [cross_entropy, accuracy]
#
#                 feed_dict_valid = {x_pl: mnist_data.validation.images, y_pl: mnist_data.validation.labels}
#                 _loss, _acc = sess.run(fetches_valid, feed_dict_valid)
#
#                 valid_loss.append(_loss)
#                 valid_accuracy.append(_acc)
#                 print(
#                     "Epoch {} : Train Loss {:6.3f}, Train acc {:6.3f},  Valid loss {:6.3f},  Valid acc {:6.3f}".format(
#                         mnist_data.train.epochs_completed, train_loss[-1], train_accuracy[-1], valid_loss[-1],
#                         valid_accuracy[-1]))
#
#         test_epoch = mnist_data.test.epochs_completed
#         while mnist_data.test.epochs_completed == test_epoch:
#             x_batch, y_batch = mnist_data.test.next_batch(batch_size)
#             feed_dict_test = {x_pl: x_batch, y_pl: y_batch}
#             _loss, _acc = sess.run(fetches_valid, feed_dict_test)
#             y_preds = sess.run(fetches=y, feed_dict=feed_dict_test)
#             y_preds = tf.argmax(y_preds, axis=1).eval()
#             y_true = tf.argmax(y_batch, axis=1).eval()
#             cm.batch_add(y_true,y_preds)
#             test_loss.append(_loss)
#             test_accuracy.append(_acc)
#         print('Test Loss {:6.3f}, Test acc {:6.3f}'.format(
#             np.mean(test_loss), np.mean(test_accuracy)))
#
#
#     except KeyboardInterrupt:
#         pass
#
#
# print(cm.accuracy())
# epoch = np.arange(len(train_loss))
# plt.figure()
# plt.plot(epoch, train_accuracy,'r', epoch, valid_accuracy,'b')
# plt.legend(['Train Acc','Val Acc'], loc=4)
# plt.xlabel('Epochs'), plt.ylabel('Acc'), plt.ylim([0.75,1.03])
# plt.show()