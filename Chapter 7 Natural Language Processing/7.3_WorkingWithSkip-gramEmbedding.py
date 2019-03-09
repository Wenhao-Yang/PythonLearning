#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 7.3_WorkingWithSkip-gramEmbedding.py
@Time: 2019/3/4 20:41
@Overview: In the recipe, we consider the order of words in creating word embedding. The first method we will explore is called skip-gram embedding. In 2013, Tomas Mikolov ang other researchers at Google authored a paper about creating word embeddings that addresses this issue, and they named their method Word2vec. The basic idea is to create word embedding that capture the relational aspect of words. And a neural network is used to predict surrounding words giving an input word.
"""
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
sess = tf.Session()

# Model parameters
batch_size = 50
embedding_size = 200
vocabulary_size = 10000
generations = 50000
print_loss_every = 500
num_sampled = int(batch_size/2)
window_size = 2
stops = stopwords.words('english')
print_valid_every = 2000
valid_words = ['cliche', 'love', 'hate', 'silly', 'sad']

# Data load function
def load_movie_data():
    save_folder_name = '../LocalData'
    save_file_name = os.path.join(save_folder_name, 'rt-polaritydata.tar.gz')
    pos_file = os.path.join(save_folder_name, 'rt-polaritydata','rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polaritydata','rt-polarity.neg')

    # if not os.path.exists(save_file_name):
    #     movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    #     stream_data = urllib.request.urlopen(movie_data_url)
    #     with open(save_file_name, 'wb') as f:
    #         f.write(stream_data.read())

    if not os.path.exists(os.path.join(save_folder_name, 'rt-polaritydata')):
        movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

        # Save tar.gz file
        req = requests.get(movie_data_url, stream=True)
        with open(save_file_name, 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
    if not (os.path.exists(pos_file) and os.path.exists(neg_file)):
        # tar_file = tarfile.open('../LocalData/rt-polaritydata.tar.gz', mode='r:gz')
        # pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
        # neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
        # pos_data = []
        # for line in pos:
        #         pos_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
        # neg_data = []
        # for line in neg:
        #         neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
        # tar_file.close()
        #
        # with open(pos_file, 'w') as pos_file_handler:
        #         pos_file_handler.write(''.join(pos_data))
        # with open(neg_file, 'w') as neg_file_handler:
        #         neg_file_handler.write(''.join(neg_data))
        # Extract tar.gz file into temp folder
        tar = tarfile.open(save_file_name, "r:gz")
        tar.extractall(path=save_folder_name)
        tar.close()

    pos_data = []
    neg_data = []
    with open(pos_file, 'r', encoding='latin-1') as temp_pos_file:
        for row in temp_pos_file:
            pos_data.append(row.encode('ascii', errors='ignore').decode())

    with open(neg_file, 'r', encoding='latin-1') as temp_neg_file:
        for row in temp_neg_file:
            neg_data.append(row.encode('ascii', errors='ignore').decode())

    neg_data = [x.rstrip() for x in neg_data]
    texts = pos_data + neg_data
    target = [1]*len(pos_data) + [0]*len(neg_data)
    return(texts, target)

texts, target = load_movie_data()
# Normalize the text
def normalize_text(texts, stops):
    texts = [x.lower() for x in texts]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    # Remove stopwords
    texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]
    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    return (texts)
texts = normalize_text(texts, stops)

# Arbitrarily set the length of reviews to three or more words
target = [target[ix] for ix,x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]

# Define a function that creates a dictionary of words with their count
def build_dictionary(sentences, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]
    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['RARE', 1]]
    # Ass most frequent words, limited to the N-most frequent
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    word_dict = {}
    # For each word, that we want in the dictionary, add it, then make it the value of the prior dictionary length
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    return (word_dict)

# Declare the function that will convert a list of sentences into lists of word indices that we can pass into our embedding lookup function.
def text_to_numbers(sentences, word_dict):
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentence:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return (data)

# Actually create the dictionary and transform the list of sentences into lists of word indices
word_dictionary = build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_to_numbers(texts, word_dictionary)

# Look up the index for the validation words that chose in step 2
valid_examples = [word_dictionary[x] for x in valid_words]

# Define a function that will return the skip-gram batches. Train on pairs of words where one word is the training input(from the target word at the center of the window) and the other word is selected from the window.
def generate_batch_data(sentences, batch_size, window_size, method='skip-gram'):
    # Full up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # Select random sentence to start
        rand_sentence = np.random.choice(sentences)
        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max((ix-window_size), 0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]
        # Denote which element of each window is the center word of interest
        label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]
        # Pull out center word of interest for each window and create a tuple for each window
        if method=='skip-gram':
            batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x, y in zip(window_sequences, label_indices)]
            # Make it into a big list of tuples(target word, surrounding word)
            tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
        else:
            raise ValueError('Method {} not implementd yet.'.format(method))

        # Extract batch and label
        batch, labels = [list(x) for x in zip(*tuple_data)]
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])

    # Trim batch and label at the enc
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    # Convet to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return (batch_data, label_data)
# Initialize the embedding matrix and declare the placeholders, and the embedding lookup function
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Lookup the word embedding
embed = tf.nn.embedding_lookup(embeddings, x_inputs)

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

# Optimizer function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)

# Train the embeddings
loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size)
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
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

plt.plot(loss_x_vec, loss_vec, 'r', label='Train Set Accuracy')
plt.title('Train set Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Generation')
plt.legend(loc='lower right')

plt.show()
