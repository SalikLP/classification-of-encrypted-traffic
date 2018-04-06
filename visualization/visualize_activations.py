import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np
from tf import dataset
from visualization import classes_module as md, vis_utils


def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")


num_headers = 8
hidden_units = 50
dir = '../../Data/h5/extracted/{0}/'.format(num_headers)
input_size = 54*num_headers
data = dataset.read_data_sets(dir, one_hot=True, validation_size=0.1, test_size=0.1, balance_classes=False,
                              payload_length=input_size, seed=0)
load_dir = "../trained_models/"
model_name = 'header_{0}_{1}_units.ckpt'.format(num_headers, hidden_units)
sess = tf.Session()
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(load_dir + model_name + ".meta")
saver.restore(sess, tf.train.latest_checkpoint(load_dir))

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
names = [tensor.name for tensor in graph.as_graph_def().node]
x_pl = graph.get_tensor_by_name("xPlaceholder:0")
layer1 = graph.get_tensor_by_name('layer1/activation:0')
W1 = graph.get_tensor_by_name("layer1/W:0")
b1 = graph.get_tensor_by_name("layer1/b:0")
W_out = graph.get_tensor_by_name("output_layer/W:0")
b_out = graph.get_tensor_by_name("output_layer/b:0")
batch_size = 20
X, T = data.test.next_batch(batch_size, shuffle=False)
classes = np.array(dataset._label_encoder.classes_)
# vis_utils.visualize(X, vis_utils.graymap, 'data.png')
l1_weights, l1_biases, out_weights, out_biases = sess.run([W1, b1, W_out, b_out])
nn = md.Network([md.Linear(l1_weights, l1_biases), md.ReLU(),
                 md.Linear(out_weights, out_biases), md.ReLU()
                 ])
plot_index = 1
sensitivity_fig = plt.figure(figsize=(20, 10))
relevance_fig = plt.figure(figsize=(20, 10))
for i, v in enumerate(X):
    t = T[i]
    classname = classes[np.argmax(t)]
    Y = np.array([nn.forward(v)])
    S = np.array([nn.gradprop(t)**2])
    D = nn.relprop(Y*t)
    s_data = vis_utils.plt_vector(S, vis_utils.heatmap,  num_headers)
    s_title = "Sensitivity for {0}".format(classname)
    rel_data = vis_utils.plt_vector(D, vis_utils.heatmap, num_headers)
    r_title = "Relevance for {0}".format(classname)
    vis_utils.add_subplot(s_data, batch_size, plot_index, s_title, sensitivity_fig)
    vis_utils.add_subplot(rel_data, batch_size, plot_index, r_title, relevance_fig)

    plot_index += 1
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for space between subplots,
               # expressed as a fraction of the average axis width
hspace = 0.2   # the amount of height reserved for space between subplots,
               # expressed as a fraction of the average axis height
# relevance_fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
#                     wspace=wspace, hspace=1.5)
plt.show()
print("Done")
# vis_utils.visualize(S, vis_utils.heatmap, 'sensitivity.png')


# units = sess.run(layer1, feed_dict={x_pl: np.reshape(stimuli, [1, 810], order='F')})


#
# # plotNNFilter(units)
# def getActivations(layer, stimuli):
#     units = sess.run(layer, feed_dict={x: np.reshape(stimuli, [1, 784], order='F'), keep_prob: 1.0})
#     plotNNFilter(units)