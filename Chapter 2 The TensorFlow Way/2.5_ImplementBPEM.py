#!/usr/bin/env python
# encoding: utf-8

"""
@author: yangwenhao
@contact: 874681044@qq.com
@software: PyCharm
@file: 2.5_ImplementBPEM.py
@time: 2018/12/9 下午3:55
@overview:We need an aggregate measure of the distance between the tow values. Here is how to change the simple
 regression algorithm from above into printing out the loss in the training loop and evaluating the loss at the end.
"""
'''
@version:
@author: yangwenhao
@contact: 874681044@qq.com
@software: PyCharm
@file: 2.2_BinaryClassification.py
@time: 2018/12/6 下午5:32
@overview: This is an example of linear regression: y = Ax. X cound be get from N(1, 0.1) and Y=10. So the A is supposed to be 10.
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#start a graph session
sess = tf.Session()

#create the data, palceholders, and the A variable
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
batch_size = 25

#Set partition
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

A = tf.Variable(tf.random_normal(shape=[1, 1]))

#add multiplication operation to the graph
my_output = tf.multiply(x_data, A)
loss = tf.reduce_mean(tf.square(my_output - y_target))

#initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

#declare a way to optimize the variables in the graph
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)

#loop through the training algorithm and tell tensorflows to train many times
for i in range(100):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i+1)%25 == 0:
        print('Step #' + str(i+1) + ' A = ' +str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

mse_test = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
mse_train = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
print('MSE on test:' + str(np.round(mse_test, 2)))
print('MSE on train: ' + str(np.round(mse_train, 2)))