import utils
import dataset
import tensorflow as tf
import tf_utils as tfu
import confusionmatrix as conf
import numpy as np
dir = '../../Data/'
# filenames = ['netflix-0602_1042',
#              'netflix-0602_1058',
#              'netflix-0602_1119',
#              'https-download']
# for filename in filenames:
#     utils.save_pcap(dir, filename)
#     print("Save done!", filename)
# filename = 'netflix-0602_1042'

#
# utils.read_pcap_raw(dir, filename)
#
data = dataset.read_data_sets(dir, one_hot=True, validation_size=10000, test_size=20000, balance_classes=True)
tf.reset_default_graph()
num_classes = len(dataset._label_encoder.classes_)

cm = conf.ConfusionMatrix(num_classes, class_names=dataset._label_encoder.classes_)
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)

x_pl = tf.placeholder(tf.float32, [None, 1460], name='xPlaceholder')
y_pl = tf.placeholder(tf.float64, [None, num_classes], name='yPlaceholder')
y_pl = tf.cast(y_pl, tf.float32)

x = tfu.ffn_layer('layer1', x_pl, 730, activation=tf.nn.relu)
x = tfu.ffn_layer('layer2', x, 730, activation=tf.nn.relu)
x = tfu.ffn_layer('layer3', x, 730, activation=tf.nn.relu)
y = tfu.ffn_layer('output_layer', x, hidden_units=num_classes, activation=tf.nn.softmax)

with tf.variable_scope('loss'):
    # computing cross entropy per sample
    cross_entropy = -tf.reduce_sum(y_pl * tf.log(y + 1e-8), reduction_indices=[1])

    # averaging over samples
    cross_entropy = tf.reduce_mean(cross_entropy)

with tf.variable_scope('training'):
    # defining our optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    # applying the gradients
    train_op = optimizer.minimize(cross_entropy)

with tf.variable_scope('performance'):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))

    # averaging the one-hot encoded vector
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Training Loop
batch_size = 100
max_epochs = 20

valid_loss, valid_accuracy = [], []
train_loss, train_accuracy = [], []
test_loss, test_accuracy = [], []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Begin training loop')

    try:
        while data.train.epochs_completed < max_epochs:
            _train_loss, _train_accuracy = [], []

            ## Run train op
            x_batch, y_batch = data.train.next_batch(batch_size)
            fetches_train = [train_op, cross_entropy, accuracy]
            feed_dict_train = {x_pl: x_batch, y_pl: y_batch}
            _, _loss, _acc = sess.run(fetches_train, feed_dict_train)

            _train_loss.append(_loss)
            _train_accuracy.append(_acc)

            ## Compute validation loss and accuracy
            if data.train.epochs_completed % 1 == 0 \
                    and data.train._index_in_epoch <= batch_size:
                train_loss.append(np.mean(_train_loss))
                train_accuracy.append(np.mean(_train_accuracy))

                fetches_valid = [cross_entropy, accuracy]

                feed_dict_valid = {x_pl: data.validation.payloads, y_pl: data.validation.labels}
                _loss, _acc = sess.run(fetches_valid, feed_dict_valid)

                valid_loss.append(_loss)
                valid_accuracy.append(_acc)
                print(
                    "Epoch {} : Train Loss {:6.3f}, Train acc {:6.3f},  Valid loss {:6.3f},  Valid acc {:6.3f}".format(
                        data.train.epochs_completed, train_loss[-1], train_accuracy[-1], valid_loss[-1],
                        valid_accuracy[-1]))

        test_epoch = data.test.epochs_completed
        while data.test.epochs_completed == test_epoch:
            x_batch, y_batch = data.test.next_batch(batch_size)
            feed_dict_test = {x_pl: x_batch, y_pl: y_batch}
            _loss, _acc = sess.run(fetches_valid, feed_dict_test)
            y_preds = sess.run(fetches=y, feed_dict=feed_dict_test)
            y_preds = tf.argmax(y_preds, axis=1).eval()
            y_true = tf.argmax(y_batch, axis=1).eval()
            cm.batch_add(y_true,y_preds)
            test_loss.append(_loss)
            test_accuracy.append(_acc)
        print('Test Loss {:6.3f}, Test acc {:6.3f}'.format(
            np.mean(test_loss), np.mean(test_accuracy)))


    except KeyboardInterrupt:
        pass


print(cm)
#
# # df = utils.load_h5(dir + filename+'.h5', key=filename.split('-')[0])
# # print("Load done!")
# # print(df.shape)
# # payloads = df['payload'].values
# # payloads = utils.pad_elements_with_zero(payloads)
# # df['payload'] = payloads
# #
# # # Converting hex string to list of int... Maybe takes to long?
# # payloads = [[int(i, 16) for i in list(x)] for x in payloads]
# # np_payloads = numpy.array(payloads)
# # # dataset = DataSet(np_payloads, df['label'].values)
# # x, y = dataset.next_batch(10)
# # batch_size = 100
# # features = {'payload': payloads}
# #
# #
# # gb = df.groupby(['ip.dst', 'ip.src', 'port.dst', 'port.src'])
# #
# #
# # l = dict(list(gb))
# #
# #
# # s = [[k, len(v)] for k, v in sorted(l.items(), key=lambda x: len(x[1]), reverse=True)]
#
#
# print("DONE")
