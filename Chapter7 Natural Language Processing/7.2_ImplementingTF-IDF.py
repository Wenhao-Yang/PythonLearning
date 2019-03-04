#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 7.2_ImplementingTF-IDF.py
@Time: 2019/3/1 15:47
@Overview: In spam prediction, since we can choose the embedding for each word, we might decide to change the weighting on certain words. One strategy is to upweight useful words and downweight overly common or too rarewords. TF-IDF is an acronym that stands for Text Frequency - Inverse Document Frequency. Here we take into consideration the word frequency.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import string
import requests
import io
import nltk
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer

sess = tf.Session()
batch_size = 200
max_features = 1000

# Load data
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

texts = [x[1] for x in text_data]
target = [x[0] for x in text_data]
# Relabel s[am as 1, ham as 0
target = [1. if x=='spam' else 0. for x in target]

# Decrease the vocabulary size
# Normalize the text
texts = [x.lower() for x in texts]
# Remove punctuation
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
# Remove numbers
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
# Trim extra whitespace
texts = [' '.join(x.split()) for x in texts]

# Tell scikt-learn's TF-IDF processing functions how to tokenize the sentences, which means how to break up a sentence into the corresponding words. Anf a great tokenizer is already built in the nltk package.
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return (words)
# Create TF-IDF of texts
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features)
sparse_tfidf_texts = tfidf.fit_transform(texts)

# Train and test sets
train_indices = np.random.choice(sparse_tfidf_texts.shape[0], round(0.8*sparse_tfidf_texts.shape[0]), replace=False)
test_indices = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(train_indices)))
texts_train = sparse_tfidf_texts[train_indices]
texts_test = sparse_tfidf_texts[test_indices]
target_train = np.array([x for ix,x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix,x in enumerate(target) if ix in test_indices])

# Declare the model
A = tf.Variable(tf.random_normal(shape=[max_features, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
x_data = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

model_output = tf.add(tf.matmul(x_data, A), b)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

prediction = tf.round(tf.sigmoid(model_output))
prediction_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(prediction_correct)

# Optimizer and initialization
my_opt = tf.train.GradientDescentOptimizer(0.0025)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

# Loop through 10000 generations
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    rand_indices = np.random.choice(texts_train.shape[0], size=batch_size)
    rand_x = texts_train[rand_indices].todense()
    rand_y = np.transpose([target_train[rand_indices]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    # Only record loss and accuracy every 100 generations
    if (i + 1) % 100 == 0:
        i_data.append(i+1)

        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(loss, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)

        temp_acc_train = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(temp_acc_train)

        temp_acc_test = sess.run(accuracy, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_acc.append(temp_acc_test)
    if (i + 1) % 500 == 0:
        acc_and_loss = [i+1, train_loss_temp, test_loss_temp, temp_acc_train, temp_acc_test]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

plt.subplot(121)
plt.plot(i_data, train_loss, 'r', label='Train Loss')
plt.plot(i_data, test_loss, 'g', label='Test Loss')
plt.title('Cross Entropy Loss per Generation')
plt.ylabel('Cross Entry Loss')
plt.xlabel('Generation')
plt.legend(loc='upper right')

plt.subplot(122)
plt.plot(i_data, train_acc, 'r', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'g', label='Test Set Accuracy')
plt.title('Train and Test set Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Generation')
plt.legend(loc='lower right')

plt.show()


