#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 5.1_WorkingWithNearestNeighbors.py
@Time: 2019/1/11 下午9:18
@Overview: We will use the Boston housing datasets and be predicting the median neighborhood housing value as a function of several features. We will find the k-NNs to the prediction points and do a weightd average of the target value.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.Session()

# Load the data using the requests module
housing_path = '../LocalData/housing.data.txt'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
num_feature = len(cols_used)

# Request and parse(解析) data
housing_file = open(housing_path).readlines()

housing_data = [[float(x) for x in y.split() if len(x)>=1] for y in housing_file if len(y)>=1]

# Sperate the data into dependent and independent features. We will predict the last variable, MEDV, which is the median value for the group of houses. We will not use the features ZN, CHAS, and RAD because of their uninformative or binary nature.
y_vals = np.transpose([np.array([y[13] for y in housing_data])])
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

x_vals = ((x_vals - x_vals.min(0)) / x_vals.ptp(0)).astype(np.float32)


# Split the x and y values into the train and test datasets. We will create the training set by selecting about 80% of the rows at random, and leave the remaining 20% for the test set
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# K value and batch size
k = 4
batch_size = len(x_vals_test)

# Declare the placeholders
x_data_train = tf.placeholder(shape=[None, num_feature], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, num_feature], dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Declare the distance function for a batch of test points. We use L1 distance here.
distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_vals_test, 1))), reduction_indices=2)

# L2: distance = tf.sqrt(tf.reduce(tf.square(tf.substract(x_data_train, tf.expand_dims(x_vals_test, 1))), reduction_indices=1))

# Create the prediction function. To do this, we use the top_k() function, which returns the values and indices of the largest values in a tensor. Since we want the indices of the smallest distances, we will instead find the k-biggest negative distances. And we also declare the predictions and the mean squared error(MSE) of the target values.
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
x_sum = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1), 1)
x_sums_repeated =tf.matmul(x_sum, tf.ones([1, k], tf.float32))
x_vals_weights = tf.expand_dims(tf.div(top_k_xvals, x_sums_repeated), 1)

top_k_yvals = tf.gather(y_target_train, top_k_indices)
prediction = tf.squeeze(tf.matmul(x_vals_weights, top_k_yvals), squeeze_dims=[1])
mse = tf.div(tf.reduce_sum(tf.square(tf.subtract(prediction, y_target_test))), batch_size)

# Test
num_loops = int(np.ceil(len(x_vals_test)/batch_size))

for i in range(num_loops):
    min_index = i*batch_size
    max_index = min((i+1)*batch_size, len(x_vals_train))
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    prediction = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test:y_batch})
    batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test:y_batch})

    print('Batch #' + str(i+1) + 'MSE: ' + str(np.round(batch_mse, 3)))

bins = np.linspace(5, 50, 45)
plt.hist(prediction, bins, alpha=0.5, label='Prediction')
plt.hist(y_batch, bins, alpha=0.5, label='Actual')
plt.title('Histogram of Predicted and Actual Values')
plt.xlabel('Med Home Value in $1,000s')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()