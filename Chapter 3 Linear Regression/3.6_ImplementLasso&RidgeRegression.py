#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 3.6_ImplementLasso&RidgeRegression.py
@Time: 2018/12/18 下午6:40
@Overview:There are also ways to limit the influence of coefficients on the regression output. These methods are called regularization methods and two of the most common regularization mathods are lasso and ridge regression. We add regularization terms to limit the slopes in the formula.

For lasso regression, we must add a term that greatly increases our loss function if the slope, A, gets above a certain value. We use a continuous approximation to a step function, called the continuous heavy step function, that is scaled up and over to the regularization cut off we choose.

For ridge regression, we just ass a term to the L2 norm, which is the scaled L2 norm of the slope coefficient.
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

# Declare the learning rate, batch size, placeholders, and variables.
learning_rate = 0.001
batch_size = 50
iterations = 1500
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target =  tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Write the formula for the linear model, y=Ax+b
model_output = tf.add(tf.matmul(x_data, A), b)

# Add the loss function, which is a modified continuous heavyside step function. We also set the cutoff for lasso regression at 0.9. This means that we want to restrict the slope coefficient to be less that 0.9.
"""
lasso_param = tf.constant(0.9)
heavyside_step = tf.truediv(1., tf.add(1., tf.exp(tf.multiply(-100., tf.subtract(A, lasso_param)))))
regularization_param = tf.multiply(heavyside_step, 99.)
loss = tf.add(tf.reduce_mean(tf.square(y_target - model_output)), regularization_param)
"""

# For the ridge regression
ridge_param = tf.constant(1.)
ridge_loss = tf.reduce_mean(tf.square(A))
loss = tf.expand_dims(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), tf.multiply(ridge_param, ridge_loss)), 0)

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
    loss_vec.append(temp_loss[0])

    if (i+1)%300==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss[0]))

[slope] = sess.run(A)
[y_intercept] = sess.run(b)
best_fit = []
for i in x_vals:
    best_fit.append(slope*i + y_intercept)

plt.plot(x_vals, y_vals, 'o')
plt.plot(x_vals, best_fit, 'r-', label='Best fit lines', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width of Ridge Regression')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

plt.plot(loss_vec, 'k-')
plt.title('Ridge Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss Value')
plt.legend(loc='upper right')
plt.show()

