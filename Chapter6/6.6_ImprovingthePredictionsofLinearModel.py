#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 6.6_ImprovingthePredictionsofLinearModel.py
@Time: 2019/2/24 下午4:31
@Overview: In the recipe, we will attempt to improve the logistic model of low birthweight with using a neural network.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
sess = tf.Session()

# Set random seed for reproducible results
seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)

# Load the data
birthweight_url = '../LocalData/lowbwt.txt'
birth_file = open(birthweight_url).readlines()
birth_data = birth_file

birth_header = [x for x in birth_data[0].split() if len(x)>=1]
birth_data = [[float(x) for x in y.split() if len(x) >=1] for y in birth_data[1:] if len(y) >= 1]
y_vals = np.array([x[1] for x in birth_data])
x_vals = np.array([x[2:9] for x in birth_data])

# Split the dataset into test and train sets
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Logistic regression convergence works better when the features are scaled between 0 and 1.
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# Declare the batch size and placeholder
batch_size = 90
x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# define init logistic functions
def init_variable(shape):
    return (tf.Variable(tf.random_normal(shape=shape)))

def logistic(input_layer, multiplicatioin_weight, bias_weight, activation = True):
    linear_layer = tf.add(tf.matmul(input_layer, multiplicatioin_weight), bias_weight)

    if activation:
        return (tf.nn.sigmoid(linear_layer))
    else:
        return (linear_layer)

# Declare 3 layers
# First logistic layer (7 input to 14 hidden nodes)
A1 = init_variable(shape=[7, 14])
b1 = init_variable(shape=[14])
logistic_layer1 = logistic(x_data, A1, b1)

# Second logistic layer (14 hiddent input to 5 hidden nodes)
A2 = init_variable(shape=[14, 5])
b2 = init_variable(shape=[5])
logistic_layer2 = logistic(logistic_layer1, A2, b2)

#Final output layer (5 hidden nodes to 1 output)
A3 = init_variable(shape=[5, 1])
b3 = init_variable(shape=[1])
final_output = logistic(logistic_layer2, A3, b3, activation=False)

# Loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output, labels=y_target))
my_opt = tf.train.AdamOptimizer(0.002)
train_step = my_opt.minimize(loss)

# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

# Create the prediction and accuracy operation on the graph
prediction = tf.round(tf.nn.sigmoid(final_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Train for 1500 iterations
loss_vec = []
train_acc = []
test_acc = []
for i in range(1500):
    rand_indices = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_indices]
    rand_y = np.transpose([y_vals_train[rand_indices]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))

    temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_acc.append(temp_acc_train)

    temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_acc.append(temp_acc_test)
    if (i + 1) % 150 == 0:
        print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss))

plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss Value')
plt.legend(loc='upper right')
plt.show()

plt.plot(train_acc, 'k-', label='Train Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.show()
