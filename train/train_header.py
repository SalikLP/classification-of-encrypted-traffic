import tensorflow as tf
from tf import tf_utils as tfu, confusionmatrix as conf, dataset, early_stopping as es
import numpy as np
import datetime
from sklearn import metrics
from sklearn.preprocessing import label_binarize
# from sklearn.metrics import roc_curve, auc
import utils
import os
from scipy import interp


def roc(y_true, y_preds, num_classes, labels, micro=True, macro=True):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_preds[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    if micro:
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_preds.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    if macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= num_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    for i in range(num_classes):
        utils.plot_class_ROC(fpr, tpr, roc_auc, i, labels)
    utils.plot_multi_ROC(fpr, tpr, roc_auc, num_classes, labels, micro, macro)


now = datetime.datetime.now()

summaries_dir = '../tensorboard'

# hidden_units = 12
units = [5]
acc_list = []
num_headers_train = []
hidden_units_train = []
num_headers = [8]
for num_header in num_headers:
    for hidden_units in units:
        hidden_units_train.append(hidden_units)
        train_dirs = ["E:/Data/LinuxChrome/{}/".format(num_header),
                      "E:/Data/WindowsSalik/{}/".format(num_header),
                      "E:/Data/WindowsAndreas/{}/".format(num_header),
                      "E:/Data/WindowsFirefox/{}/".format(num_header)
                      ]

        test_dirs = ["E:/Data/WindowsChrome/{}/".format(num_header)]

        trainstr = "train:"
        for traindir in train_dirs:
            trainstr += traindir.split('Data/')[1].split("/")[0]
            trainstr += ":"
        teststr = "test:"
        for testdir in test_dirs:
            teststr += testdir.split('Data/')[1].split("/")[0]
            teststr += ":"
        timestamp = "%.2d%.2d_%.2d%.2d" % (now.day, now.month, now.hour, now.minute)
        save_dir = "../trained_models/{0}/{1}/{2}/".format(num_header, hidden_units, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        seed = 0
        namestr = trainstr+teststr+str(num_header)+":"+str(hidden_units)
        # Beta for L2 regularization
        beta = 1.0
        val_size = [0.1]

        early_stop = es.EarlyStopping(patience=20, min_delta=0.05)
        subdir = "/{0}/{1}/{2}/".format(num_header, hidden_units, timestamp)
        input_size = num_header*54
        data = dataset.read_data_sets(test_dirs, train_dirs, merge_data=True, one_hot=True,
                                      validation_size=0.1,
                                      test_size=0.1,
                                      balance_classes=False,
                                      payload_length=input_size,
                                      seed=seed)
        tf.reset_default_graph()
        num_headers_train.append(num_header)
        num_classes = len(dataset._label_encoder.classes_)
        labels = dataset._label_encoder.classes_

        # cm = conf.ConfusionMatrix(num_classes, class_names=labels)
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)

        x_pl = tf.placeholder(tf.float32, [None, input_size], name='xPlaceholder')
        y_pl = tf.placeholder(tf.float64, [None, num_classes], name='yPlaceholder')
        y_pl = tf.cast(y_pl, tf.float32)

        x = tfu.ffn_layer('layer1', x_pl, hidden_units, activation=tf.nn.relu, seed=seed)
        # x = tf.layers.dense(x_pl, hidden_units, tf.nn.relu)
        # x = tfu.dropout(x, 0.5)
        # x = tfu.ffn_layer('layer2', x, hidden_units, activation=tf.nn.relu)
        # x = tfu.ffn_layer('layer2', x, 50, activation=tf.nn.sigmoid)
        # x = tfu.ffn_layer('layer3', x, 730, activation=tf.nn.relu)
        y = tfu.ffn_layer('output_layer', x, hidden_units=num_classes, activation=tf.nn.softmax, seed=seed)
        y_ = tf.argmax(y, axis=1)
        # with tf.name_scope('cross_entropy'):
        #   # The raw formulation of cross-entropy,
        #   #
        #   # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #   #                               reduction_indices=[1]))
        #   #
        #   # can be numerically unstable.
        #   #
        #   # So here we use tf.losses.sparse_softmax_cross_entropy on the
        #   # raw logit outputs of the nn_layer above.
        #   with tf.name_scope('total'):
        #     cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_pl, logits=y)


        with tf.variable_scope('loss'):
            # computing cross entropy per sample
            cross_entropy = -tf.reduce_sum(y_pl * tf.log(y + 1e-8), reduction_indices=[1])

            W1 = tf.get_default_graph().get_tensor_by_name("layer1/W:0")
            loss = tf.nn.l2_loss(W1)
            # averaging over samples
            cross_entropy = tf.reduce_mean(cross_entropy)
            loss = tf.reduce_mean(cross_entropy + loss * beta)

        tf.summary.scalar('cross_entropy', cross_entropy)

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

        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir + '/train/' + subdir)
        val_writer = tf.summary.FileWriter(summaries_dir + '/validation/' + subdir)

        # Training Loop
        batch_size = 100
        max_epochs = 200

        valid_loss, valid_accuracy = [], []
        train_loss, train_accuracy = [], []
        test_loss, test_accuracy = [], []
        epochs = []
        with tf.Session() as sess:
            early_stop.on_train_begin()
            train_writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())
            total_parameters = 0
            print("Calculating trainable parameters!")
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                print("Shape: {}".format(shape))
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                print("Shape {0} gives {1} trainable parameters".format(shape, variable_parameters))
                total_parameters += variable_parameters
            print("Trainable parameters: {}".format(total_parameters))
            print('Begin training loop')
            saver = tf.train.Saver()
            try:
                while data.train.epochs_completed < max_epochs:
                    _train_loss, _train_accuracy = [], []
                    ## Run train op
                    x_batch, y_batch = data.train.next_batch(batch_size)
                    fetches_train = [train_op, cross_entropy, accuracy, merged]
                    feed_dict_train = {x_pl: x_batch, y_pl: y_batch}
                    _, _loss, _acc, _summary = sess.run(fetches_train, feed_dict_train)

                    _train_loss.append(_loss)
                    _train_accuracy.append(_acc)
                    ## Compute validation loss and accuracy
                    if data.train.epochs_completed % 1 == 0 \
                            and data.train._index_in_epoch <= batch_size:

                        train_writer.add_summary(_summary, data.train.epochs_completed)

                        train_loss.append(np.mean(_train_loss))
                        train_accuracy.append(np.mean(_train_accuracy))

                        fetches_valid = [cross_entropy, accuracy, merged]

                        feed_dict_valid = {x_pl: data.validation.payloads, y_pl: data.validation.labels}
                        _loss, _acc, _summary = sess.run(fetches_valid, feed_dict_valid)
                        epochs.append(data.train.epochs_completed)
                        valid_loss.append(_loss)
                        valid_accuracy.append(_acc)
                        val_writer.add_summary(_summary, data.train.epochs_completed)
                        current = valid_loss[-1]
                        early_stop.on_epoch_end(data.train.epochs_completed, current)
                        print("Epoch {} : Train Loss {:6.3f}, Train acc {:6.3f},  Valid loss {:6.3f},  Valid acc {:6.3f}"
                              .format(data.train.epochs_completed, train_loss[-1], train_accuracy[-1], valid_loss[-1],
                                      valid_accuracy[-1]))
                        if early_stop.stop_training:
                            early_stop.on_train_end()
                            break


                test_epoch = data.test.epochs_completed
                while data.test.epochs_completed == test_epoch:
                    batch_size = 1000
                    x_batch, y_batch = data.test.next_batch(batch_size)
                    feed_dict_test = {x_pl: x_batch, y_pl: y_batch}
                    _loss, _acc, _summary = sess.run(fetches_valid, feed_dict_test)
                    y_preds = sess.run(fetches=y, feed_dict=feed_dict_test)
                    y_preds = tf.argmax(y_preds, axis=1).eval()
                    y_true = tf.argmax(y_batch, axis=1).eval()
                    # cm.batch_add(y_true, y_preds)
                    test_loss.append(_loss)
                    test_accuracy.append(_acc)
                    # test_writer.add_summary(_summary, data.train.epochs_completed)
                print('Test Loss {:6.3f}, Test acc {:6.3f}'.format(
                    np.mean(test_loss), np.mean(test_accuracy)))
                namestr += ":acc{:.3f}".format(np.mean(test_accuracy))
                acc_list.append("{:.3f}".format(np.mean(test_accuracy)))
                saver.save(sess, save_dir+'header_{0}_{1}_units.ckpt'.format(num_header, hidden_units))
                feed_dict_test = {x_pl: data.test.payloads, y_pl: data.test.labels}
                y_preds = sess.run(fetches=y, feed_dict=feed_dict_test)
                y_true = data.test.labels
                roc(y_true, y_preds, num_classes, labels, micro=False, macro=False)
                y_preds = sess.run(fetches=y_, feed_dict=feed_dict_test)
                y_true = tf.argmax(data.test.labels, axis=1).eval()
                y_true = [labels[i] for i in y_true]
                y_preds = [labels[i] for i in y_preds]
                conf = metrics.confusion_matrix(y_true, y_preds, labels=labels)
                report = metrics.classification_report(y_true, y_preds, labels=labels)
                nostream_dict = ['http', 'https']
                y_stream_true = []
                y_stream_preds = []
                for i, v in enumerate(y_true):
                    pred = y_preds[i]
                    if v in nostream_dict:
                        y_stream_true.append('non-streaming')
                    else:
                        y_stream_true.append('streaming')
                    if pred in nostream_dict:
                        y_stream_preds.append('non-streaming')
                    else:
                        y_stream_preds.append('streaming')
                stream_acc = len([v for i, v in enumerate(y_stream_preds) if v == y_stream_true[i]]) / len(
                    y_stream_true)

                # Binarize the output
                y_stream_true1 = label_binarize(y_stream_true, classes=['non-streaming', 'streaming'])
                y_stream_preds1 = label_binarize(y_stream_preds, classes=['non-streaming', 'streaming'])
                n_classes = y_stream_true1.shape[1]

                roc(y_stream_true1, y_stream_preds1, n_classes, ['non-streaming', 'streaming'], micro=False, macro=False)










                conf1 = metrics.confusion_matrix(y_true, y_preds, labels=labels)
                conf2 = metrics.confusion_matrix(y_stream_true, y_stream_preds, labels=['non-streaming', 'streaming'])
                report = metrics.classification_report(y_true, y_preds, labels=labels)
                report2 = metrics.classification_report(y_stream_true, y_stream_preds, labels=['non-streaming', 'streaming'])

            except KeyboardInterrupt:
                pass
        print(namestr)
        utils.plot_confusion_matrix(conf1, labels, save=False, title=namestr)
        utils.plot_confusion_matrix(conf2, ['non-streaming', 'streaming'], save=False,
                                    title="StreamNoStream_acc{}".format(stream_acc))
        print(report)
        print(report2)
        # utils.plot_metric_graph(x_list=epochs, y_list=valid_accuracy, save=False, x_label="epochs", y_label="Accuracy", title="Accuracy")
        # utils.plot_metric_graph(x_list=epochs, y_list=valid_loss, save=False,  x_label="epochs", y_label="loss", title="Loss")
        # utils.plot_confusion_matrix(conf, labels, save=False, title=namestr)
        utils.show_plot()

acc_list = list(map(float, acc_list))
print(acc_list, hidden_units_train)


