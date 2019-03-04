#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 3.7_ImplementElasticNetRegression.py
@Time: 2018/12/20 下午2:17
@Overview: Elastic net regression is a type of regression that combines lasso regression with ridge regression by adding a L1 and L2 regularization term to the loss function. And we will implement this in multiple linear regression on the irirs dataset, instead of sticking to the two-dimensional data as before. We use pedal length, pedal width, and sepal width to predict sepal length,
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
x_vals = np.array([[x[1], x[2], x[3]] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# Declare the learning rate, batch size, placeholders, and variables.
learning_rate = 0.001
batch_size = 50
iterations = 1000
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[3, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
model_output = tf.add(tf.matmul(x_data, A), b)

# For elastic net, the loss function has the L1 and L2 norms of the partial slopes. We create these terms and then add them into the loss funtion.
elastic_param1 = tf.constant(1.)
elastic_param2 = tf.constant(1.)
l1_a_loss = tf.reduce_mean(tf.abs(A))
l2_a_loss = tf.reduce_mean(tf.square(A))
e1_term = tf.multiply(elastic_param1, l1_a_loss)
e2_term = tf.multiply(elastic_param2, l2_a_loss)
loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)

# Initialize the variables, declare the optimizer, and loop through training part
init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

loss_vec = []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss[0])

    if (i+1)%250==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss[0]))

plt.plot(loss_vec, 'k-')
plt.title('Elastic Net Regression Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss Value')
plt.legend(loc='upper right')
plt.show()
