#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 5.5_UsingNearestNeighborForImageRecognition.py
@Time: 2019/1/24 下午9:25
@Overview: We use the MNIST dataset for Image recognition. And it's composed of thousands of labeled images that are 28*28 pexels in size. And we could compare the results to a neural network in the later chapter. We import the Python Image Library to be able to plot a sample of the predicted outputs.
"""
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

# Load the MINST dataset in a one hot encoded form
sess = tf.Session()
mnist = input_data.read_data_sets("../LocalData/MNIST_data/", one_hot=True)

# Because the MNIST dataset is large and computing the distances between 784 features on tens of thousands of imputs would be computationally hard, we will sample a smaller set of images to train on. Also, we choose a test set number that is divisible by six six only for plotting purposes, as we will plot the last batch of six images to see a sample of results.
train_size = 1000
test_size = 102
rand_train_indices= np.random.choice(len(mnist.train.images), train_size, replace=False)
rand_test_indices = np.random.choice(len(mnist.test.images), test_size, replace=False)
x_vals_train = mnist.train.images[rand_train_indices]
x_vals_test = mnist.test.images[rand_test_indices]
y_vals_train = mnist.train.labels[rand_train_indices]
y_vals_test = mnist.test.labels[rand_test_indices]

# Declare the k value and batch size
k = 4
batch_size = 6

# Initialize the placeholder
x_data_train = tf.placeholder(shape=[None, 784], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None, 10], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# Distance matrix(L1)
distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), reduction_indices=2)

# L2:
# distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), reduction_indices=1))

# Find the top k images that are the closest and predict the mode. The mode will be performed on one hot encoded indices and counting which occurs the most.
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
prediction_indices = tf.gather(y_target_train, top_k_indices)
count_of_predictions = tf.reduce_sum(prediction_indices, reduction_indices=1)
prediction = tf.argmax(count_of_predictions, dimension=1)

# Loop thought our test set, compute the predictions and store them
num_loops = int(np.ceil(len(x_vals_test)/batch_size))
test_output = []
actual_vals = []
for i in range(num_loops):
    min_index = i*batch_size
    max_index = min((i+1)*batch_size, len(x_vals_train))
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})
    test_output.extend(predictions)
    actual_vals.extend(np.argmax(y_batch, axis=1))

# Calaulate the accuracy
accuracy = sum([1./test_size for i in range(test_size) if test_output[i]==actual_vals[i]])
print('Accuracy on test set:' + str(accuracy))

# Plot the last batch result
actuals = np.argmax(y_batch, axis=1)
Nrows = 2
Ncols = 3
for i in range(len(actuals)):
    plt.subplot(Nrows, Ncols, i+1)
    plt.imshow(np.reshape(x_batch[i], [28, 28]), cmap='Greys_r')
    plt.title('Actual:' + str(actuals[i]) + 'Prediction: ' + str(predictions[i]), fontsize=10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
plt.show()