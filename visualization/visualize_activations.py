import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tf import dataset
from visualization import classes_module as md, vis_utils

num_headers = 16
hidden_units = 12
train_dirs = ['C:/Users/salik/Documents/Data/LinuxChrome/{0}/'.format(num_headers),
                'C:/Users/salik/Documents/Data/WindowsFirefox/{0}/'.format(num_headers),
                'C:/Users/salik/Documents/Data/WindowsAndreas/{0}/'.format(num_headers),
                'C:/Users/salik/Documents/Data/WindowsSalik/{0}/'.format(num_headers)]

test_dirs = ['C:/Users/salik/Documents/Data/WindowsChrome/{0}/'.format(num_headers)]
seed = 0
input_size = 54*num_headers
data = dataset.read_data_sets(train_dirs, test_dirs, merge_data=True, one_hot=True,
                                  validation_size=0.1,
                                  test_size=0.1,
                                  balance_classes=False,
                                  payload_length=input_size,
                                  seed=seed)
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
y_pl = graph.get_tensor_by_name("yPlaceholder:0")
layer1 = graph.get_tensor_by_name('layer1/activation:0')
W1 = graph.get_tensor_by_name("layer1/W:0")
b1 = graph.get_tensor_by_name("layer1/b:0")
W_out = graph.get_tensor_by_name("output_layer/W:0")
b_out = graph.get_tensor_by_name("output_layer/b:0")
y = graph.get_tensor_by_name("output_layer/activation:0")
# Get index of prediction
y_ = tf.argmax(y, axis=1)

feed_dict = {x_pl: data.test.payloads, y_pl: data.test.labels}
y_preds = sess.run(fetches=y_, feed_dict=feed_dict)
y_true = tf.argmax(data.test.labels, axis=1).eval(session=sess)

# Number of samples to plot
sample_size = 10
num_samples_picked = 0
sample_payloads = []
sample_labels = []
classes = np.array(dataset._label_encoder.classes_)
# Which class do we want to visualise
class_name = "drtv"
class_number = np.argmax(classes == class_name)
# Get all the places where that class label is the true label
true_idx = [i for i, v in enumerate(y_true) if v == class_number]
# What predictions did the network make at those indicies
preds_at_idx = y_preds[true_idx]
# Iterate over predictions and pick #sample_size where prediction was right
for i, v in enumerate(preds_at_idx):
    if v == class_number and num_samples_picked < sample_size:
        sample_payloads.append(data.test.payloads[true_idx[i]])
        sample_labels.append(data.test.labels[true_idx[i]])
        num_samples_picked += 1


X = np.array(sample_payloads)
T = np.array(sample_labels)
# Extract the weights and biases from the network
l1_weights, l1_biases, out_weights, out_biases = sess.run([W1, b1, W_out, b_out])
# Create visualisation network in sequential manner
nn = md.Network([md.Linear(l1_weights, l1_biases), md.ReLU(),
                 md.Linear(out_weights, out_biases), md.ReLU()
                 ])
# Make a sensitivity and relevance plot for each sample
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
    vis_utils.plot_data(s_data, s_title)
    vis_utils.plot_data(rel_data, r_title)
plt.show()
