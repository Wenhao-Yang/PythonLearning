#!/usr/bin/env python
# encoding: utf-8

"""
@version: 
@author: yangwenhao
@contact: 874681044@qq.com
@software: PyCharm
@file: 2.2_BinaryClassification.py
@time: 2018/12/6 下午5:32
@overview: This is an example of binary classification. And one set of samples is from N(-1,1). And another set is from N(3,1).
"""

import numpy as np
import tensorflow as tf

sess = tf.Session()

#create data from two different normal distribution, N(-1, 1) and N(3, 1)
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

#next add the translation operation to the graph.
my_output = tf.add(x_data, A)

#because the specific loss function expects batches of data that have an extra dimension , add an extra dimension to the output with the function expand_dims()
my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

#initilize variables
init = tf.initialize_all_variables()
sess.run(init)

#declare loss function which is cross entropy with unscaled logits taht transforms thenm with a sigmoid function
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expanded, labels=y_target_expanded)

#add an optimizer function to the graph
my_opt = tf.train.GradientDescentOptimizer(learning_rate=.05)
train_step = my_opt.minimize(xentropy)

#loop through a randomly selected data point
for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%200 == 0:
        print('Step # ' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))


