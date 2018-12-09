#!/usr/bin/env python
# encoding: utf-8

"""
@author: yangwenhao
@contact: 874681044@qq.com
@software: PyCharm
@file: 2.3_BatchAndStochastic.py
@time: 2018/12/6 下午8:31
@overview:Extend the prior regression example using stochastic training to batch training

"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#start a graph session
sess1 = tf.Session()

#declare a batch size
batch_size = 20

#create the data, palceholders, and the A variable. We cahnge the shape of placeholders. They are two dimensions, where the first dimension is None(we could explicitly set it to 20), and the second will be the number of data points in the batch.
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))

#add matrix multiplication operation to the graph
my_output = tf.matmul(x_data, A)

#add loss function between the multiplication output and the target data. The loss function will change bexause we have to take the mean of all L2 losses of each data point in the batch.
loss = tf.reduce_mean(tf.square(my_output - y_target))

#initialize all variables
init = tf.initialize_all_variables()
sess1.run(init)

#declare a way to optimize the variables in the graph
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)

#loop through the training algorithm and tell tensorflows to train many times. initialize a list to store the loss function every fivce intervals
loss_batch = []
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess1.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i+1)%5 == 0:
        print('Step #' + str(i+1) + ' A = ' + str(sess1.run(A)))
        temp_loss = sess1.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)

#start a graph session
sess2 = tf.Session()

#create the data, palceholders, and the A variable
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1]))

#add multiplication operation to the graph
my_output = tf.multiply(x_data, A)

#add loss function between the multiplication output and the target data
loss = tf.square(my_output - y_target)

#initialize all variables
init = tf.initialize_all_variables()
sess2.run(init)

#declare a way to optimize the variables in the graph
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)


loss_stochastic = []
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess2.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i+1)%5 == 0:
        print('Step #' + str(i+1) + ' A = ' + str(sess2.run(A)))
        temp_loss = sess2.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_stochastic.append(temp_loss)

plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label="Stochastic Loss")
plt.plot(range(0, 100, 5), loss_batch, 'r--', label="Batch Loss, size=20")
plt.legend(loc='upper right', prop={'size': 11})

plt.show()