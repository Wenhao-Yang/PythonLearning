'''
@version:
@author: yangwenhao
@contact: 874681044@qq.com
@software: PyCharm
@file: 2.2_BinaryClassification.py
@time: 2018/12/6 下午5:32
@overview: This is an example of linear regression: y = Ax. X cound be get from N(1, 0.1) and Y=10. So the A is supposed to be 10.
'''

import numpy as np
import tensorflow as tf

#start a graph session
sess = tf.Session()

#create the data, palceholders, and the A variable
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1]))

#add multiplication operation to the graph
my_output = tf.multiply(x_data, A)

#add loss function between the multiplication output and the target data
loss = tf.square(my_output - y_target)

#initialize all variables
init = tf.initialize_all_variables()
sess.run(init)

#declare a way to optimize the variables in the graph
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)

#loop through the training algorithm and tell tensorflows to train many times
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i+1)%25 == 0:
        print('Step #' + str(i+1) + ' A = ' +str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))