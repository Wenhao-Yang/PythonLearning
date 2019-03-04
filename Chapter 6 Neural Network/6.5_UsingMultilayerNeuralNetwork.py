#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 6.5_UsingMultilayerNeuralNetwork.py
@Time: 2019/2/22 上午10:20
@Overview: In the recipe, we apply a multilayer neural network with different layers to predict birthweight in the Low Birthweight dataset.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
sess = tf.Session()

# Load the data through local file
birthweight_url = '../LocalData/lowbwt.txt'
birth_file = open(birthweight_url).readlines()
birth_data = birth_file

birth_header = [x for x in birth_data[0].split() if len(x)>=1]
birth_data = [[float(x) for x in y.split() if len(x) >=1] for y in birth_data[1:] if len(y) >= 1]
y_vals = np.array([x[10] for x in birth_data])
clos_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI', 'FTV']

x_vals = np.array([[x[ix] for ix,feature in enumerate(birth_header) if feature in clos_of_interest] for x in birth_data])

print(x_vals.size)

# Set random seed
seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)
batch_size = 100

# Split the dataset into test and train sets
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Normalize input feature to be between 0 and 1
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# Initialize variables
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape=shape, stddev=st_dev))
    return (weight)

def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape=shape, stddev=st_dev))
    return (bias)

x_data = tf.placeholder(shape=[None, 8], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Fully connected layer function
def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return (tf.nn.relu(layer))

# The model has hidden layers of sizes 25, 10 and 3

# First Layer
weight_1 = init_weight(shape=[8, 25], st_dev=[10.0])
bias_1 = init_bias(shape=[25], st_dev=[10.0])
layer_1 = fully_connected(x_data, weight_1, bias_1)

# Second Layer
weight_2 = init_weight(shape=[25, 10], st_dev=[10.0])
bias_2 = init_bias(shape=[10], st_dev=[10.0])
layer_2 = fully_connected(layer_1, weight_2, bias_2)

# Third Layer
weight_3 = init_weight(shape=[10, 3], st_dev=[10.0])
bias_3 = init_bias(shape=[3], st_dev=[10.0])
layer_3 = fully_connected(layer_2, weight_3, bias_3)

# Forth Layer
weight_4 = init_weight(shape=[3, 1], st_dev=[10.0])
bias_4 = init_bias(shape=[1], st_dev=[10.0])
final_output = fully_connected(layer_3, weight_4, bias_4)

# Loss function and optimizer
loss = tf.reduce_mean(tf.abs(y_target - final_output))
my_opt = tf.train.AdamOptimizer(0.05)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# Train for 200 iteration
loss_vec = []
test_loss = []

for i in range(200):
    # First we select a random set of indices for the batch
    rand_indices = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_indices]
    rand_y = np.transpose([y_vals_train[rand_indices]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(test_temp_loss)
    if (i+1)%25==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))

plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Compared the train and test accuracy with Logistic Regression, we could find that we achieved around 60% accuracy after 1000 iterations. And here by creating an indicatorif they are above or below 2500 grams, the following code shows the acccuracy of Neural Network way.
actuals = np.array([x[1] for x in birth_data])
test_actuals = actuals[test_indices]
train_actuals = actuals[train_indices]
test_pres = [x[0] for x in sess.run(final_output, feed_dict={x_data: x_vals_test})]
train_pres = [x[0] for x in sess.run(final_output, feed_dict={x_data: x_vals_train})]
test_pres = np.array([1.0 if x<2500.0 else 0.0 for x in test_pres])
train_pres = np.array([1.0 if x<2500.0 else 0.0 for x in train_pres])

test_acc = np.mean([x==y for x,y in zip(test_pres, test_actuals)])
train_acc = np.mean([x==y for x,y in zip(train_pres, train_actuals)])
print('On predicting the category of low birthweight from regression output (<2500g):')
print('Test Acc:{}'.format(test_acc))
print('Train acc:{}'.format(train_acc))




