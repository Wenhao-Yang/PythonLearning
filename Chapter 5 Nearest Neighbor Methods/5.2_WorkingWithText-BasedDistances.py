#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 5.2_WorkingWithText-BasedDistances.py
@Time: 2019/1/15 下午5:20
@Overview: We implement the Nearest Neighbor method based on text-based distances with TensorFlow. The text-based distance is the Levenshtein distance(the edit distance) between strings. The levenshtein distance is the minimal number of edits to get from one string to another string. The allowed edits are inserting a character, deleting a character, or substituting a character with a different one.
"""
import tensorflow as tf
sess = tf.Session()

# The way of calculating the edit distance between two words.
hypothesis = list('bear')
truth = list('beers')
h1 = tf.SparseTensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]], hypothesis, [1, 1, 1])
t1 = tf.SparseTensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]], truth, [1, 1, 1])

print('The Levenshtein Distance Between \'bear\' and \'beers\' is '+ str(sess.run(tf.edit_distance(h1, t1, normalize=False))))

# The way of comparing two words, bear and beer, both with another word, beers.
hypothesis2 = list('bearbeer')
truth2 = list('beersbeers')
h2 = tf.SparseTensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]], hypothesis2, [1, 2, 4])
t2 = tf.SparseTensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3],[0, 0, 4], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4]], truth2, [1, 2, 5])

print('The Levenshtein Distance Between \'bearbeer\' and \'beersbeers\' is '+ str(sess.run(tf.edit_distance(h2, t2, normalize=True))))

# Compare a set of words against another word.
hypothesis_words = ['bear', 'bar', 'tensor', 'flow']
truth_word = ['beers']
num_h_words = len(hypothesis_words)
h_indices = [[xi, 0, yi] for xi,x in enumerate(hypothesis_words) for yi,y in enumerate(x)]
h_chars = list(''.join(hypothesis_words))
h3 = tf.SparseTensor(h_indices, h_chars, [num_h_words, 1, 1])
truth_word_vec = truth_word*num_h_words
t_indices = [[xi, 0, yi] for xi,x in enumerate(truth_word_vec) for yi,y in enumerate(x)]
t_chars = list(''.join(truth_word_vec))
t3 = tf.SparseTensor(t_indices, t_chars, [num_h_words, 1, 1])

print('The Levenshtein Distance Between a set of word and \'beers\' is \n' + str(sess.run(tf.edit_distance(h3, t3, normalize=True))))

# Calculate the distance using placeholders
def create_sparse_vec(word_list):
    num_words = len(word_list)
    indices = [[xi, 0, yi] for xi,x in enumerate(word_list) for yi,y in enumerate(x)]
    chars = list(''.join(word_list))
    return(tf.SparseTensorValue(indices, chars, [num_words, 1, 1]))

hyp_string_sparse = create_sparse_vec(hypothesis_words)
truth_string_sparse = create_sparse_vec(truth_word*len(hypothesis_words))

hyp_input = tf.sparse_placeholder(dtype=tf.string)
truth_input = tf.sparse_placeholder(dtype=tf.string)

edit_distances = tf.edit_distance(hyp_input, truth_input, normalize=True)

feed_dict = {hyp_input: hyp_string_sparse, truth_input: truth_string_sparse}

print('The Levenshtein Distance Calculated by Placeholders, Between a set of word and \'beers\' is \n' + str(sess.run(edit_distances, feed_dict=feed_dict)))