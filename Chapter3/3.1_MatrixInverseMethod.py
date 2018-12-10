#!/usr/bin/env python
# encoding: utf-8

"""
@author: yangwenhao
@contact: 874681044@qq.com
@software: PyCharm
@file: 3.1_MatrixInverseMethod.py
@time: 2018/12/10 下午7:46
@overview:Using TensorFlow to solve two dimensional linear regressions with the matrix inverse method
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.Session()

#create data, x_vals: evenly spaced numbers over a specified interval;
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)

#create the A matrix first, which will be a column of x-data and a cloumn of 1s.
x_vals_colmun = np.transpose(np.matrix(x_vals))
ones_column  = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_colmun, ones_column))
b = np.transpose(np.matrix(y_vals))

#turn the A and b matrix into tensors
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

#use TensorFlow to solve this via the matrix inverse method
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
solution = tf.matmul(product, b_tensor)
solution_eval = sess.run(solution)

#extract the coeddicients from the solution, the slope and the y-intercept
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
print('slope: ' + str(slope))
print('y_intercept: ' + str(y_intercept))

best_fit = []
for i in x_vals:
    best_fit.append(slope*i + y_intercept)
plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit lines', linewidth=3)
plt.legend(loc='upper left')
plt.show()

