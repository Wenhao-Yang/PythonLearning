#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 7.4_WorkingWithCBOWEmbedding.py
@Time: 2019/3/9 20:08
@Overview: Implement the CBOW method of word2vec. Predict a single target word from a surrounding window of context words.
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

# Set Random Seeds
tf.set_random_seed(42)
np.random.seed(42)

# Declare model parameters
batch_size = 500
embedding_size = 200
vocabulary_size = 2000
generations = 50000
model_learning_rate = 0.25
num_sampled = int(batch_size/2)
window_size = 3
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 100
stops = stopwords.words('english')
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']

texts, target = text_helpers.load_movie_data()
texts = text_helpers.normalize_text(texts, stops)

# Texts must contain at least 3 words
target = [target[ix] for ix,x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]

# Declare the vocabulary dictionary
word_dictionary = text_helpers.build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_helpers.text_to_numbers(texts, word_dictionary)

valid_examples = [word_dictionary[x] for x in valid_words]

# Initialize the embedding matrix and declare the placeholders, and the embedding lookup function
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

x_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2*window_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Lookup the word embedding
embed = tf.zeros([batch_size, embedding_size])
for element in range(2*window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

# Use a loos function called noise-contrastive error to avoid problems caused by the sparse categories results.
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=y_target,
                                     inputs=embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))

# Create a way to find nearby words to our validation words. Compute the cosine similarity between the validation set and all of our word embedding
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Use the tensorflow train.Saver method to save teh embeddings
saver = tf.train.Saver({"embeddings": embeddings})

# Optimizer function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate).minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)

# Filter out sentences that aren't long enough:
text_data = [x for x in text_data if len(x)>=(2*window_size+1)]

# Train the embeddings
loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, batch_size, window_size, method='cbow')
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}
    sess.run(optimizer, feed_dict=feed_dict)

    if (i+1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {} : {}'.format(i+1, loss_val))

    if (i+1) % print_valid_every == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_words = word_dictionary_rev[valid_examples[j]]
            top_k = 5
            nearest = (-sim[j, :]).argsort()[1:top_k+1]
            log_str = 'Nearest to {}:'.format(valid_words)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = '{} {}'.format(log_str, close_word)
            print(log_str)

    # Save dictionary + embeddings
    if (i+1) % save_embeddings_every == 0:
        # Save vocabulary dictionary
        with open(os.path.join('temp', 'movie_vocab.pkl'), 'wb') as f:
            pickle.dump(word_dictionary, f)

        # save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(), 'temp', 'cbow_movie_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file:{}'.format(save_path))

plt.plot(loss_x_vec, loss_vec, 'r', label='Train Set Loss')
plt.title('Train Loss Values')
plt.ylabel('LOss')
plt.xlabel('Generation')
plt.legend(loc='upper right')

plt.show()
