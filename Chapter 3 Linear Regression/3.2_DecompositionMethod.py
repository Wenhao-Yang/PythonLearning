#!/usr/bin/env python
# encoding: utf-8

"""
@author: yangwenhao
@contact: 874681044@qq.com
@software: PyCharm
@file: 3.2_DecompositionMethod.py
@time: 2018/12/11 下午5:07
@overview: We implement a matrix decomposition method for linear method. Implementing inverse methods in the previous recipe can be numerically inefficient in most cases, especially when the matrices get cery large. Another approach is to use the Cholesky decomposition method. The Cholesky decomposition decomposes a mtrix into a lower and upper triangular matrix, say L and L' , such that these matrices are transposition of each other. Here we solve the system, Ax=b, by writing it as LL'x=b. We will first solve Ly=b and then solve L'x=y to arrive at our coefficient matrix ,x.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()

#Create the data, and obtain the A and b matrix in the same way as before.
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))

A = np.column_stack((x_vals_column, ones_column))
b = np.transpose(np.matrix(y_vals))

A_tensor = tf.constant(A)
b_tensot = tf.constant(b)

#Find the Cholesky decomposition of our square matrix
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
L = tf.cholesky(tA_A)
tA_b = tf.matmul(tf.transpose(A_tensor), b)
sol1 = tf.matrix_solve(L, tA_b)
sol2 = tf.matrix_solve(tf.transpose(L), sol1)

#Extract the coefficients
solution_eval = sess.run(sol2)
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
print('Slope: ' + str(slope))
print('y_intercept: ' + str(y_intercept))

best_fit = []
for i in x_vals:
    best_fit.append(slope*i + y_intercept)

#The code could be used for image size modification
# plt.figure(figsize=(10.8, 7.2))
plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit lines', linewidth=3)
plt.legend(loc='upper left')
plt.title('Solving Linear Regression with Cholesky Decomposition Method', fontsize=12)
plt.show()
