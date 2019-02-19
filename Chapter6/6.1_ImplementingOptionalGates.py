#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 6.1_ImplementingOptionalGates.py
@Time: 2019/1/25 上午9:20
@Overview: One of the most fundamental concepts of neural networks is an operation known as an operation gate. Here, we will start with a multiplication operation as a gate and then we will consider a nested gate operation. TensorFlow keeps track of our model's operations and variable values and makes adjustments in respect of our optimization algorithm specification and the output of the loss function.
"""
import tensorflow as tf
sess = tf.Session()

# Variables and Operations
a = tf.Variable(tf.constant(4.))
x_vals = 5.
x_data = tf.placeholder(dtype=tf.float32)
multiplication = tf.multiply(a, x_data)

# L2 distance as the loss function
loss = tf.square(tf.subtract(multiplication, 50.))

# Initialize the model
init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Optimize the model
print('Optimizing a Multiplication Gate Output to 50.')
for i in range(10):
    sess.run(train_step, feed_dict={x_data: x_vals})
    a_val = sess.run(a)
    mult_output = sess.run(multiplication, feed_dict={x_data: x_vals})

    print(str(a_val) + '*' + str(x_vals) + '=' + str(mult_output))

from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.Session()

a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
x_vals = 5.
x_data = tf.placeholder(dtype=tf.float32)

two_gate = tf.add(tf.multiply(a, x_data), b)

loss = tf.square(tf.subtract(two_gate, 50.))

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

print('\nOptimizing Two Gate Output to 50.')
for i in range(10):
    # Run the train steps
    sess.run(train_step, feed_dict={x_data: x_vals})
    # Get the a abd b
    a_val, b_val = (sess.run(a), sess.run(b))
    # Run the two-gate graph output
    two_gate_output = sess.run(two_gate, feed_dict={x_data: x_vals})

    print(str(a_val) + '*' + str(x_vals) + '+' + str(b_val) + '=' + str(two_gate_output))