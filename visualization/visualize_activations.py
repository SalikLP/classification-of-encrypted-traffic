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


dir = '../../Data/h5/extracted/'
input_size = 810
data = dataset.read_data_sets(dir, one_hot=True, validation_size=0.1, test_size=0.1, balance_classes=False,
                              payload_length=input_size)
load_dir = "../trained_models/"
model_name = "header_50_units.ckpt"
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
X, T = data.test.next_batch(1)
# vis_utils.visualize(X, vis_utils.graymap, 'data.png')
l1_weights, l1_biases, out_weights, out_biases = sess.run([W1, b1, W_out, b_out])
# X, T = vis_utils.getMNISTsample(N=12, path='C:\\Users\\Salik\\Documents\\classification-of-encrypted-traffic', seed=1234)
nn = md.Network([md.Linear(l1_weights, l1_biases), md.ReLU(),
                 md.Linear(out_weights, out_biases), md.ReLU()
                 ])
Y = nn.forward(X)
S = nn.gradprop(T)**2
D = nn.relprop(Y*T)

vis_utils.plt_vector(S, vis_utils.heatmap, "Sensitivity")
vis_utils.plt_vector(D, vis_utils.heatmap, "Relevance")
plt.show()
print("Done")
# vis_utils.visualize(S, vis_utils.heatmap, 'sensitivity.png')


# units = sess.run(layer1, feed_dict={x_pl: np.reshape(stimuli, [1, 810], order='F')})



# plotNNFilter(units)
def getActivations(layer, stimuli):
    units = sess.run(layer, feed_dict={x: np.reshape(stimuli, [1, 784], order='F'), keep_prob: 1.0})
    plotNNFilter(units)