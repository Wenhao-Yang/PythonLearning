#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 6.4_ImplementDifferentLayers.py
@Time: 2019/2/20 下午12:17
@Overview:Here we create and use convolutional layers and maxpool layers with fully connected data.
"""
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
sess = tf.Session()

data_size = 25
data_1d = np.random.normal(size=data_size)
x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])

# Make a convolutional layer and decalre a random filter
def conv_layer_1d(input_1d, my_filter):
    # Make 1d input into 4d
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform convolution
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,1,1,1], padding='VALID')
    #Drop extra dimensions
    conv_output_1d = tf.squeeze(convolution_output)
    return (conv_output_1d)

my_filter = tf.Variable(tf.random_normal(shape=[1,5,1,1]))
my_convolution_output = conv_layer_1d(x_input_1d, my_filter)

# Create the activaiton fucntion
def activation(input_1d):
    return(tf.nn.relu(input_1d))

my_activation_output = activation(my_convolution_output)

# Declare the maxpool layer function
def max_pool(input_1d, width):
    # Make 1d input into 4d
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1], strides=[1,1,1,1], padding='VALID')
    pool_output_1d = tf.squeeze(pool_output)
    return (pool_output_1d)

my_maxpool_output = max_pool(my_convolution_output, width=5)

# Fully connected layers
def fully_connected(input_layer, num_output):
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_output]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_output])

    input_layer_2d = tf.expand_dims(input_layer, 0)
    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)

    full_output_1d = tf.squeeze(full_output)
    return (full_output_1d)

my_full_output = fully_connected(my_maxpool_output, 5)

# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)
feed_dict={x_input_1d: data_1d}

print('Input = array of length 25')
print('Convolution, filter, lenght = 5, stride size = 1, results in an array of length 21:')
print(sess.run(my_convolution_output, feed_dict=feed_dict))

print('\nInput = the above array of length 21')
print('ReLU element wise returns the array of length 21:')
print(sess.run(my_activation_output, feed_dict=feed_dict))

print('\nInput = the above array of length 21')
print('MaxPool, window length = 5, stride size = 1, results in the array of lenght 17:')
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

print('\nInput = the above array of length 17')
print('Fully connected layer on all four rows with five outputs:')
print(sess.run(my_full_output, feed_dict=feed_dict))

ops.reset_default_graph()
sess = tf.Session()

data_size = [10, 10]
data_2d = np.random.normal(size=data_size)
x_input_2d = tf.placeholder(dtype=tf.float32, shape=data_size)

# Convolution function
def conv_layer_2d(input_2d, my_filter):
    # First, change 2d input to 4d
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1, 2, 2, 1], padding='VALID')
    conv_output_2d = tf.squeeze(convolution_output)

    return (conv_output_2d)

my_filter = tf.Variable(tf.random_normal(shape=[2, 2, 1, 1]))
my_convolution_output = conv_layer_2d(x_input_2d, my_filter)

# Activation Function
def activation(input_2d):
    return (tf.nn.relu(input_2d))
my_activation_output = activation(my_convolution_output)

# MaxPool
def max_pool(input_2d, width, height):
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, height, width, 1], strides=[1, 1, 1, 1], padding='VALID')

    pool_output_2d = tf.squeeze(pool_output)
    return (pool_output_2d)
my_maxpool_output = max_pool(my_activation_output, width=2, height=2)

# Fully connected layer
def fully_connected(input_layer, num_output):
    flat_input = tf.reshape(input_layer, [-1])
    weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input), [num_output]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_output])

    input_layer_2d = tf.expand_dims(flat_input, 0)
    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)

    full_output_2d = tf.squeeze(full_output)
    return (full_output_2d)

my_full_output = fully_connected(my_maxpool_output, 5)

# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)
feed_dict={x_input_2d: data_2d}

print('\nInput = [10 * 10] array ')
print('2 * 2 Convolution, stride size = 2 * 2, results in an [5 * 5] array:')
print(sess.run(my_convolution_output, feed_dict=feed_dict))

print('\nInput = the above [5 * 5] array ')
print('ReLU element wise returns the [5 * 5] array ')
print(sess.run(my_activation_output, feed_dict=feed_dict))

print('\nInput = the above [5 * 5] array')
print('MaxPool, stride size = [2 * 2], results in a [4 * 4] array:')
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

print('\nInput = the above [4 * 4] array')
print('Fully connected layer on all four rows with five outputs:')
print(sess.run(my_full_output, feed_dict=feed_dict))