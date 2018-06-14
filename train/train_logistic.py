from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tf import tf_utils as tfu, dataset, early_stopping as es
import numpy as np
import datetime
from sklearn import metrics
import utils
import operator
from sklearn.metrics import classification_report
import pandas as pd
from collections import Counter

def input_fn(data):
    return (data.payloads, data.labels)


now = datetime.datetime.now()

num_headers = 16
hidden_units = 15
train_dirs = ['/home/mclrn/Data/WindowsSalik/{0}/'.format(num_headers),
              '/home/mclrn/Data/WindowsChrome/{0}/'.format(num_headers),
              '/home/mclrn/Data/WindowsLinux/{0}/'.format(num_headers),
              '/home/mclrn/Data/WindowsFirefox/{0}/'.format(num_headers)]

test_dirs = ['/home/mclrn/Data/WindowsAndreas/{0}/'.format(num_headers)]

val = 0.1
input_size = num_headers * 54
seed = 0
train_size = []
data = dataset.read_data_sets(train_dirs, test_dirs, merge_data=True, one_hot=False,
                              validation_size=val,
                              test_size=0.1,
                              balance_classes=False,
                              payload_length=input_size,
                              seed=seed)

train_size.append(len(data.train.payloads))
num_classes = len(dataset._label_encoder.classes_)
labels = dataset._label_encoder.classes_

classifier = LogisticRegression(verbose=2)

classifier.fit(data.train.payloads, data.train.labels)

score = classifier.score(data.test.payloads, data.test.labels)

predict_proba = classifier.predict_proba(data.test.payloads)

predict = classifier.predict(data.test.payloads)

correct = list(map(operator.sub,predict,data.test.labels)).count(0)


confusion = metrics.confusion_matrix(data.test.labels, predict)
utils.plot_confusion_matrix(confusion, labels, save=True, title="Logistic regression confusion matrixacc")
accuracy = correct/len(predict)

report = classification_report(data.test.labels, predict)
print(accuracy)
skleranacc = metrics.accuracy_score(data.test.labels,predict)
print("Accuracy calculated by sklearn.metrics: {}".format(skleranacc))
most_occurences = Counter(data.test.labels).most_common(1)
print("Naive classifier that just guesses the most frequents label will obtain an accuracy of: %s" % ((most_occurences[0][1])/len(data.test.labels)))
print(report)