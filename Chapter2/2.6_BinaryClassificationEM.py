#!/usr/bin/env python
# encoding: utf-8

"""
@author: yangwenhao
@contact: 874681044@qq.com
@software: PyCharm
@file: 2.6_BinaryClassificationEM.py
@time: 2018/12/9 下午4:47
@overview: This is an example of binary classification. And one set of samples is
from N(-1,1). And another set is from N(3,1). Create own accuracy function at the end.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.Session()

#create data from two different normal distribution, N(-1, 1) and N(3, 1)
batch_size = 25
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1, None], dtype=tf.float32)
y_target = tf.placeholder(shape=[1, None], dtype=tf.float32)

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

#next add the translation operation to the graph.
my_output = tf.add(x_data, A)

#initilize variables
init = tf.initialize_all_variables()
sess.run(init)

#declare loss function which is cross entropy with unscaled logits taht transforms thenm with a sigmoid function
xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target))

#add an optimizer function to the graph
my_opt = tf.train.GradientDescentOptimizer(learning_rate=.05)
train_step = my_opt.minimize(xentropy)

#loop through a randomly selected data point
for i in range(1800):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = [x_vals_train[rand_index]]
    rand_y = [y_vals_train[rand_index]]

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%600 == 0:
        print('Step # ' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))

y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_data, A))))
correct_prediction = tf.equal(y_prediction, y_target)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_value_test = sess.run(accuracy, feed_dict={x_data: [x_vals_test], y_target: [y_vals_test]})
acc_value_train = sess.run(accuracy, feed_dict={x_data: [x_vals_train], y_target: [y_vals_train]})

print('Accuracy on train set:' + str(acc_value_train))
print('Accuracy on test set:' + str(acc_value_test))

A_result = sess.run(A)
bins = np.linspace(-5, 5, 50)
plt.hist(x_vals[0:50], bins, alpha=0.5, label='N(-1, 1)', color='green')
plt.hist(x_vals[50:100] , bins[0:50], alpha=0.5, label='N(3, 1)', color='red')
plt.plot((A_result, A_result), (0, 8), 'k--', linewidth=3, label='A = ' + str(np.round(A_result, 2)))
plt.legend(loc='upper right')
plt.title('Bianry Classifier, Accuracy = ' + str(np.round(acc_value_test, 2)))
plt.show()
