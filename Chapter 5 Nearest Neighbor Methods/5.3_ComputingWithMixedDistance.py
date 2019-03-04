#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 5.3_ComputingWithMixedDistance.py
@Time: 2019/1/22 上午10:19
@Overview: It is important to extend the nearest neighbor algorithm to take into account variables that are scaled differently. Here we will show how to scale the distance funtion for different variables. The key to weighting the distance function is to use a weight diagonal weight matrix.
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

# Scale the x values to be between 0 and 1 with min-max scaling
x_vals = ((x_vals - x_vals.min(0)) / x_vals.ptp(0)).astype(np.float32)

# Create the diagonal weight matrix.
weight_diagonal = x_vals.std(0)
weight_matrix = tf.cast(tf.diag(weight_diagonal), dtype=tf.float32)

# Split the dataset
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

# Declare the distance function
substraction_term = tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))
first_product = tf.matmul(substraction_term, tf.tile(tf.expand_dims(weight_matrix, 0), [batch_size, 1, 1]))
second_product = tf.matmul(first_product, tf.transpose(substraction_term, perm=[0, 2, 1]))
distance = tf.sqrt(tf.matrix_diag_part(second_product))

# Return the top  K-NNs with top_k function
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
x_sum = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1), 1)
x_sums_repeated =tf.matmul(x_sum, tf.ones([1, k], tf.float32))
x_vals_weights = tf.expand_dims(tf.div(top_k_xvals, x_sums_repeated), 1)

top_k_yvals = tf.gather(y_target_train, top_k_indices)
prediction = tf.squeeze(tf.matmul(x_vals_weights, top_k_yvals), squeeze_dims=[1])

# MSE
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

















