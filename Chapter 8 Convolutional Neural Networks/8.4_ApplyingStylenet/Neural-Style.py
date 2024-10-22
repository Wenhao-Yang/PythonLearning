#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Neural-Style.py
@Time: 2019/4/8 21:20
@Overview: stylenet is a procedure that attempts to learn an image style from one picture and apply it to a second picture while keeping the second image structure.
"""
import os
import scipy.misc
import scipy.io as sio
import numpy as np
import tensorflow as tf

sess = tf.Session()
original_image_file = '../temp/pexels-photo-41120.jpeg'
style_image_file = '../temp/Die_Heuernte.jpg'

# Local parameters
vgg_path = '../temp/imagenet-vgg-verydeep-19.mat'
original_image_weight = 5.0
style_image_weight = 200.0
regularization_weight = 50.0
learning_rate = 0.1
generations = 10000
output_generations = 500

# Load the images with scipy
original_image = scipy.misc.imread(original_image_file)
style_image = scipy.misc.imread(style_image_file)

# Get shape of target and make the style image the same
target_shape = original_image.shape
style_image = scipy.misc.imresize(style_image, target_shape[1] / style_image.shape[1])

# Define the layers
vgg_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
              'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
              'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
              'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
              'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4']
# Extract parameters from the mat file
def extract_net_info(path_to_params):
    vgg_data = sio.loadmat(path_to_params)
    normalization_matrix = vgg_data['normalization'][0][0][0]
    mat_mean = np.mean(normalization_matrix, axis=(0, 1))
    network_weights = vgg_data['layers'][0]
    return(mat_mean, network_weights)

#Define the network
def vgg_network(network_weights, init_image):
    network = {}
    image = init_image
    for i, layer in enumerate(vgg_layers):
        if layer[1] == 'c':
            weights, bias = network_weights[i][0][0][0][0]
            weights = np.transpose(weights, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            conv_layer = tf.nn.conv2d(image, tf.constant(weights), (1, 1, 1, 1), 'SAME')
            image = tf.nn.bias_add(conv_layer, bias)
        elif layer[1]=='r':
            image = tf.nn.relu(image)
        else:
            image = tf.nn.max_pool(image, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
        network[layer] = image
    return(network)

# assign intermediate layers to the original and style images
original_layer = 'relu4_2'
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

# Get wieghts and mean. Change the images shape to have four dimensions
normalization_mean, network_weights = extract_net_info(vgg_path)
shape = (1,) + original_image.shape
style_shape = (1,) + style_image.shape
original_features = {}
style_features = {}

# Image placeholder
image = tf.placeholder(dtype=tf.float32, shape=shape)
vgg_net = vgg_network(network_weights, image)

# Normalize the original image matrix and run it through the network
original_minus_mean = original_image - normalization_mean
original_norm = np.array([original_minus_mean])
original_features[original_layer] = sess.run(vgg_net[original_layer], feed_dict={image: original_norm})

# Repeat the same procedure with each of the style layers
image = tf.placeholder(dtype=tf.float32, shape=style_shape)
vgg_net = vgg_network(network_weights, image)
style_minus_mean = style_image - normalization_mean
style_norm = np.array([style_minus_mean])
for layer in style_layers:
    layer_output = sess.run(vgg_net[layer], feed_dict={image: style_norm})
    layer_output = np.reshape(layer_output, (-1, layer_output.shape[3]))
    style_gram_matrix = np.matmul(layer_output.T, layer_output) / layer_output.size
    style_features[layer] = style_gram_matrix

# To create the combined image, start with random noise and run it through the network
initial = tf.random_normal(shape) * 0.05
image = tf.Variable(initial)
vgg_net = vgg_network(network_weights, image)

# Declare the first loss, on the original image.
original_loss = original_image_weight * (2 * tf.nn.l2_loss(vgg_net[original_layer] - original_features[original_layer]) / original_features[original_layer].size)

# Declare the same type of loss for each style layer
style_loss = 0
style_losses = []
for style_layer in style_layers:
    layer = vgg_net[style_layer]
    feats, height, width, channels = [x.value for x in layer.get_shape()]
    size = height * width * channels
    features = tf.reshape(layer, (-1, channels))
    style_gram_matrix = tf.matmul(tf.transpose(features), features) / size
    style_expected = style_features[style_layer]
    style_losses.append(2 * tf.nn.l2_loss(style_gram_matrix - style_expected) / style_expected.size)
style_loss += style_image_weight * tf.reduce_sum(style_losses)

# The third loss term is the total variation loss
total_var_x = sess.run(tf.reduce_prod(image[:, 1:, :].get_shape()))
total_var_y = sess.run(tf.reduce_prod(image[:, :, 1:, :].get_shape()))
first_term = regularization_weight * 2
second_term_numerator = tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1]-1, :, :])
second_term = second_term_numerator / total_var_y
third_term = (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2]-1, :]) / total_var_x)
total_variation_loss = first_term * (second_term + third_term)

# The total loss
loss = original_loss +style_loss + total_variation_loss

# Declare optimizer and training step
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(loss)
sess.run(tf.global_variables_initializer())

#Loop through the training generations and print a status update every so often and save the temporary image
for i in range(generations):
    sess.run(train_step)
    # Print update and save temporary output
    if (i+1) % output_generations == 0:
        print('Generation {} out of {]'.format(i + 1, generations))
        image_eval = sess.run(image)
        best_iamge_add_mean = image_eval.reshape(shape[1:]) + normalization_mean
        output_file = 'temp_output_{}.jpg'.format(i)
        scipy.misc.imsave(output_file, best_iamge_add_mean)
image_eval  = sess.run(image)
best_iamge_add_mean = image_eval.reshape(shape[1:]) + normalization_mean
output_file = 'final_output.jpg'
scipy.misc.imsave(output_file, best_iamge_add_mean)