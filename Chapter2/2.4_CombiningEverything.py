#!/usr/bin/env python
# encoding: utf-8

"""
@author: yangwenhao
@contact: 874681044@qq.com
@software: PyCharm
@file: 2.4_CombiningEverything.py
@time: 2018/12/9 下午2:29
@overview:Create a classifier on the iris dataset

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
sess = tf.Session()

#Load the iris data, and transform the target data to be just 1 or 0. We only use 2 features, petal lenght and petal width. These 2 features are the third and forth entry in each x_value.
iris = datasets.load_iris()
binary_target = np.array([1. if x==0 else 0. for x in iris.target])
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

#  c     eclare the batch size, data placeholers, and model variables
batch_size = 20
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

#Define the linear model: y=Ax+b
my_mult = tf.multiply(x2_data, A)
my_add = tf.add(my_mult, b)
my_output = tf.subtract(x1_data, my_add)

#Add the sigmoid cross-entropy loss function
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target)

#Declare a optimizing method
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_step = my_opt.minimize(xentropy)

#Init variables
init = tf.initialize_all_variables()
sess.run(init)

#Train the model with 1000 iterations
for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})
    if (i+1)%200==0:
        print('Step #' + str(i+1) + 'A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))

#plots the result
[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)
x = np.linspace(0, 3, num=50)
ablinValues = []

for i in x:
    ablinValues.append(slope*i + intercept)

setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==1]
setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==1]

non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==0]
non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==0]

plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa''')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='NOn-setosa')
plt.plot(x, ablinValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.title('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Pental Width')
plt.legend(loc='lower right')
plt.show()

