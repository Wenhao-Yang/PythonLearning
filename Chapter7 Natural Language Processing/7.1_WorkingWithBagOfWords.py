#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 7.1_WorkingWithBagOfWords.py
@Time: 2019/2/27 下午10:07
@Overview:In the recipe, the code shows how to work with a bag of words embedding in TensorFLow. We will use this type of embedding to do spam prediction.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
import string
import requests
import io
from zipfile import ZipFile
from tensorflow.contrib import learn
sess = tf.Session()

save_file_name = os.path.join('../', 'LocalData/temp_spam_data.csv')
text_data = []
if os.path.isfile(save_file_name):
    with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
            if len(row)==2:
                text_data.append(row)
else:
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')

    #Format data
    text_data = file.decode()
    print(text_data)

    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x)>=1]

    with open(save_file_name, 'w') as temp_output_file:
        writer = csv.writer(temp_output_file)
        writer.writerows(text_data)

print('The number of record line: ' + str(text_data.__len__()))

texts = [x[1] for x in text_data]
target = [x[0] for x in text_data]

# Relabel s[am as 1, ham as 0
target = [1 if x=='spam' else 0 for x in target]

# Normalize the text
texts = [x.lower() for x in texts]
# Remove punctuation
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
# Remove numbers
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
# Trim extra whitespace
texts = [' '.join(x.split()) for x in texts]

# Plot histogram of text lengths
text_lengths = [len(x.split()) for x in texts]
text_lengths = [x for x in text_lengths if x<50]
plt.hist(text_lengths, bins=25)
plt.title('Histogram of # of Words in Texts')
plt.show()
    
sentence_size = 25
min_word_freq = 3

# TensorFlow has a built-in processing tool for determining vocabulary , called VocabularyProcessor().
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)
vocab_processor.fit_transform(texts)
embedding_size = len(vocab_processor.vocabulary_)
print('Embedding size: ' + str(embedding_size))

# Split the data into train and test dataset
train_indicees = np.random.choice(len(texts), round(len(texts)*0.8), replace=False)
test_indices = np.array(list(set(range(len(texts))) - set(train_indicees)))
texts_train = [x for ix,x in enumerate(texts) if ix in train_indicees]
texts_test = [x for ix,x in enumerate(texts) if ix in test_indices]
target_train = [x for ix,x in enumerate(target) if ix in train_indicees]
target_test = [x for ix,x in enumerate(target) if ix in test_indices]

# Declare the embedding matrix for the words. Sentence words will be translated into indices. These indices will be translated into one-hot-encoded vectors that we can create with an identity matrix, which will be the size of our word embeddings. We will use the matrix to look up the sparse vector for each word and add them together for the sparse sentence vector.
identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

# Since we will do logistic regression to predict the probability of spam, we need to declare the logistic regression variables. Note that x_data input placeholder should be of integer type because of the row index of the identity matrix
A = tf.Variable(tf.random_normal(shape=[embedding_size, 1 ]))
b = tf.Variable(tf.random_normal(shape=[1, 1,]))
x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)

# Use the TensorFlow's embedding lookup function that will map the indices of words in the sentence to the one-hot-encode vectors of our identity matrix. When we have that matrix, we create the sentence vector by summing up the aforementioned word vectors.
x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
x_col_sums = tf.reduce_sum(x_embed, 0)

# Given the fixed-length sentence vectors for each sentence, we could perform logistic regression.
x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)

# Loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
prediction = tf.sigmoid(model_output)
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

# Init the variables
init = tf.global_variables_initializer()
sess.run(init)

# Iteration over the sentences
loss_vec = []
train_acc_all = []
train_acc_avg = []
print('Starting Training Over ' + str(len(texts_train)) +' Sentences:')
for ix,t in enumerate(vocab_processor.fit_transform(texts_train)):
    # print('No.' + str(ix) + 'vocab: ' + str(t))
    y_data = [[target_train[ix]]]

    sess.run(train_step, feed_dict={x_data: t, y_target: y_data})

    temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
    loss_vec.append(temp_loss)

    if (ix+1)%1000==0:
        print('Train Observation # ' + str(ix+1) + ': Loss = ' + str(temp_loss))

    # Keep training average of past 50 observations accuracy and Fet prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    # Get True/False if prediction is accurate
    train_acc_temp = target_train[ix]==np.round(temp_pred)
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all)>=50:
        train_acc_avg.append(np.mean(train_acc_all[-50:]))

print('Starting Test For ' + str(len(texts_test)) +' Sentences:')
test_acc_all = []
for ix,t in enumerate(vocab_processor.fit_transform(texts_test)):
    # print('No.' + str(ix) + 'vocab: ' + str(t))
    y_data = [[target_test[ix]]]

    if (ix+1)%500==0:
        print('Test Observation #' + str(ix+1))

    # Keep training average of past 50 observations accuracy and Fet prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    # Get True/False if prediction is accurate
    test_acc_temp = target_train[ix]==np.round(temp_pred)
    test_acc_all.append(test_acc_temp)
print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))




