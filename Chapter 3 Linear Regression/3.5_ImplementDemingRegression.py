#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 3.5_ImplementDemingRegression.py
@Time: 2018/12/16 下午1:42
@Overview: In the recipe, we will implement Deming regression, which means we will need a different way to measure the distance between the model line and data points.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow.python.framework import ops
import tensorflow as tf

ops.reset_default_graph()
sess = tf.Session()

# Import Iris dataset
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# Declare the learning rate, batch size, placeholders, and variables
learning_rate = 0.1
batch_size = 50
iterations = 250
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target =  tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Write the formula for the linear model, y=Ax+b
model_output = tf.add(tf.matmul(x_data, A), b)

# The loss function is a geometric formula that comprises of a numerator and denominator
deming_numerator = tf.abs(tf.subtract(y_target, tf.add(tf.matmul(x_data, A), b)))
deming_denominator = tf.sqrt(tf.add(tf.square(A), 1))
loss = tf.reduce_mean(tf.truediv(deming_numerator, deming_denominator))

# Initialize the variables, declare the optmizer, and loop through training part
init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

loss_vec = []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    if (i+1)%25==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

[slope] = sess.run(A)
[y_intercept] = sess.run(b)
best_fit = []
for i in x_vals:
    best_fit.append(slope*i + y_intercept)

plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit lines', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

plt.plot(loss_vec, 'k-', label='L1 Loss')
plt.title('Deming Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss Value')
plt.legend(loc='upper right')
plt.show()
