#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 7.6_WorkingWithDoc2vecEmbedding.py
@Time: 2019/3/14 19:21
@Overview: We extend the methodologies of train word embeddings to have a document embeddings. And it should capture the relationship of words to the document that they come from. Doc2vec simply adds an additional embedding matrix for the documents and uses a window of words to plus the document index to predict the next word. In the recipe, we concatenate the document embeddings to the end of the word embeddings.
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

# Load data and transform the text data from the text_helper
data_folder_name = 'temp'
texts, target = text_helpers.load_movie_data()

# Model parameters
embedding_size = 200
doc_embedding_size = 100
concateanted_size = embedding_size + doc_embedding_size
vocabulary_size = 7500
batch_size = 500
generations = 10000
model_learning_rate = 0.025
num_sampled = int(batch_size/2)
window_size = 3

save_embeddings_every = 5000
print_loss_every = 100
print_valid_every = 5000
stops = stopwords.words('english')
valid_words = ['love', 'hate', 'happy', 'sad','man', 'woman']

print('Normalizing Text Data')
texts = text_helpers.normalize_text(texts, stops)
target = [target[ix] for ix,x in enumerate(texts) if len(x.split())>window_size]
texts = [x for x in texts if len(x.split())>window_size]
assert(len(target)==len(texts))

print('Building the vocabulary dictionary')
word_dictionary = text_helpers.build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_helpers.text_to_numbers(texts, word_dictionary)

valid_examples = [word_dictionary[x] for x in valid_words]

# Define the word embeddings and document embeddings, and noise-contrastive loss parameters
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
doc_embeddings = tf.Variable(tf.random_uniform([len(texts), doc_embedding_size], -1.0, 1.0))

# NCE loss parameters
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, concateanted_size], stddev=1.0 / np.sqrt(concateanted_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

x_inputs = tf.placeholder(tf.int32, shape=[batch_size, window_size + 1])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Create the embedding function that adds together the word embeddings and then concatenates the document embedding at the end
embed = tf.zeros([batch_size, embedding_size])
for element in range(window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])
doc_indices = tf.slice(x_inputs, [0, window_size], [batch_size, 1])
doc_embed = tf.nn.embedding_lookup(doc_embeddings, doc_indices)
# Concatenate embeddings
final_embed = tf.concat(axis=1, values=[embed, tf.squeeze(doc_embed)])

# Loss and optimizer
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=y_target,
                                     inputs=final_embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)
train_step = optimizer.minimize(loss)

# Create a way to find nearby words to our validation words. Compute the cosine similarity between the validation set and all of our word embedding
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Use the tensorflow train.Saver method to save teh embeddings
saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings})
init = tf.global_variables_initializer()
sess.run(init)

# Iteration
# Train the embeddings
loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, batch_size, window_size, method='doc2vec')
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}
    sess.run(train_step, feed_dict=feed_dict)

    if (i+1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {} : {}'.format(i+1, loss_val))

    # Print random words and top 5 related words
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
        with open(os.path.join('temp', 'movie_doc2vec_vocab.pkl'), 'wb') as f:
            pickle.dump(word_dictionary, f)

        # save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(), 'temp', 'doc2vec_movie_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file:{}'.format(save_path))

plt.plot(loss_x_vec, loss_vec, 'r', label='Train Set Loss')
plt.title('Train Loss Values')
plt.ylabel('LOss')
plt.xlabel('Generation')
plt.legend(loc='upper right')

plt.show()