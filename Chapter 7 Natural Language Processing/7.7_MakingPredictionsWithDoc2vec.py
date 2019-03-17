#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 7.7_MakingPredictionsWithDoc2vec.py
@Time: 2019/3/17 17:40
@Overview:
"""
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import string
import requests
import collections
import io
import urllib.request
import tarfile
from nltk.corpus import stopwords
import text_helpers
sess = tf.Session()

# Model parameters
embedding_size = 200
vocabulary_size = 7500
doc_embedding_size = 100
concatenated_size = embedding_size + doc_embedding_size
logistic_batch_size = 500
max_words = 20
window_size = 3
stops = stopwords.words('english')

# Load data and transform the text data from the text_helper
data_folder_name = 'temp'
texts, target = text_helpers.load_movie_data()

print('Normalizing Text Data')
texts = text_helpers.normalize_text(texts, stops)
target = [target[ix] for ix,x in enumerate(texts) if len(x.split())>window_size]
texts = [x for x in texts if len(x.split())>window_size]

train_indices = np.sort(np.random.choice(len(target), round(0.8*len(target)), replace=False))
test_indices = np.sort(np.array(list(set(range(len(target))) - set(train_indices))))
texts_train = [x for ix,x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix,x in enumerate(texts) if ix in test_indices]

target_train = np.array([x for ix,x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix,x in enumerate(target) if ix in test_indices])

# Load the dictionary
dict_file = os.path.join(data_folder_name, 'movie_doc2vec_vocab.pkl')
word_dictionary = pickle.load(open(dict_file, 'rb'))

# Convert the loaded sentence data to a numerical numpy array with the word dictionary
text_data_train = np.array(text_helpers.text_to_numbers(texts_train, word_dictionary))
text_data_test = np.array(text_helpers.text_to_numbers(texts_test, word_dictionary))

# Standardize the reviews to be all the same length
text_data_train = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_train]])
text_data_test = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_test]])

# Model variables and placeholders
log_x_inputs = tf.placeholder(shape=[None, max_words + 1], dtype=tf.int32)
log_y_target = tf.placeholder(tf.int32, shape=[None, 1])
A = tf.Variable(tf.random_normal(shape=[concatenated_size, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Restore the prior-trained embeddings
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
doc_embeddings = tf.Variable(tf.random_uniform([len(texts), doc_embedding_size], -1.0, 1.0))

log_embed = tf.zeros([logistic_batch_size, embedding_size])
for element in range(max_words):
    log_embed += tf.nn.embedding_lookup(embeddings, log_x_inputs[:, element])

log_doc_indices = tf.slice(log_x_inputs, [0, max_words], [logistic_batch_size, 1])

# Put the embedding lookup function on the graph and take the advantage embeddings of all the words in the sentence
log_doc_embed = tf.nn.embedding_lookup(doc_embeddings, log_doc_indices)
log_final_embed = tf.concat(axis=1, values=[log_embed, tf.squeeze(log_doc_embed)])

# Model operation and loss function
model_output = tf.add(tf.matmul(log_final_embed, A), b)
log_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.cast(log_y_target, tf.float32)))

# Add prediction and accuracy function
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, tf.cast(log_y_target, tf.float32)), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Optimizer function and initialize the model varialbes
logistic_opt = tf.train.AdadeltaOptimizer(0.01)
logistic_train_step = logistic_opt.minimize(log_loss)
init = tf.global_variables_initializer()
sess.run(init)

# Tell the saver method to load the prior CBOW embeddings into the embedding variable
model_checkpoint_path = os.path.join(data_folder_name, 'doc2vec_movie_embeddings.ckpt')
saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings})
saver.restore(sess, model_checkpoint_path)

# Iteration
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    rand_indices = np.random.choice(text_data_train.shape[0], size=logistic_batch_size)
    rand_x = text_data_train[rand_indices]

    # Append review index at the end of text data
    rand_x_doc_indices = train_indices[rand_indices]
    rand_x = np.hstack((rand_x, np.transpose([rand_x_doc_indices])))

    rand_y = np.transpose([target_train[rand_indices]])

    feed_dict = {log_x_inputs: rand_x, log_y_target: rand_y}


    sess.run(logistic_train_step, feed_dict=feed_dict)

    # Only record loss and accuracy every 100 generations
    if (i + 1) % 100 == 0:
        i_data.append(i + 1)

        rand_index_test = np.random.choice(text_data_test.shape[0], size=logistic_batch_size)
        rand_x_test = text_data_test[rand_index_test]

        # Append review index at the end of text data
        rand_x_doc_indices_test = test_indices[rand_index_test]
        rand_x_test = np.hstack((rand_x_test, np.transpose([rand_x_doc_indices_test])))

        rand_y_test = np.transpose([target_test[rand_index_test]])

        test_feed_dict = {log_x_inputs: rand_x_test, log_y_target: rand_y_test}

        train_loss_temp = sess.run(log_loss, feed_dict=feed_dict)
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(log_loss, feed_dict=test_feed_dict)
        test_loss.append(test_loss_temp)

        temp_acc_train = sess.run(accuracy, feed_dict=feed_dict)
        train_acc.append(temp_acc_train)

        temp_acc_test = sess.run(accuracy, feed_dict=test_feed_dict)
        test_acc.append(temp_acc_test)

    if (i + 1) % 500 == 0:
        acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, temp_acc_train, temp_acc_test]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(
            *acc_and_loss))

# Plot loss over time
plt.plot(i_data, train_loss, 'k--', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entry Loss')
plt.legend(loc='upper right')
plt.show()

#Plot train and test accuracy
plt.plot(i_data, train_acc, 'k--', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()