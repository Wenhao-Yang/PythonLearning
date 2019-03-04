#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 5.4_UsingAddressMatchingExample.py
@Time: 2019/1/22 下午4:41
@Overview: Combining distances between observations that have both text and numerical feature. And we will match the non-typo addresses with actual street addresses.
"""
import random
import string
import numpy as np
import tensorflow as tf

# Create the reference dataset. To show succinct output, we will only make each dataset comprise of 10 addresses
n = 10
street_names = ['abbey', 'baker', 'cannal', 'donner', 'elm']
street_types = ['rd', 'st', 'ln', 'pass', 'ave']
rand_zip = [random.randint(65000, 65999) for i in range(5)] # randomly selected 5 zip numbers
numbers = [random.randint(1, 9999) for i in range(n)] # randomly selected door numbers
streets = [random.choice(street_names) for i in range(n)] # Randomly selected street names
street_suffs = [random.choice(street_types) for i in range(n)] # Randomly selected street suffs
zips = [random.choice(rand_zip) for i in range(n)]
full_streets = [str(x) + ' ' + y + ' ' + z for x,y,z in zip(numbers, streets, street_suffs)]
reference_data = [list(x) for x in zip(full_streets, zips)]

# To create the test set, we declare a function that will randomly create a typo(打印错误概率0.75) in a string and return the resulting string
def create_typo(s, prob=0.75):
    if random.uniform(0, 1) < prob:
        rand_ind = random.choice(range(len(s)))
        s_list = list(s)
        s_list[rand_ind] = random.choice(string.ascii_lowercase)
        s = ''.join(s_list)
    return s

typo_streets = [create_typo(x) for x in streets]
typo_full_streets = [str(x) + ' ' + y + ' ' + z for x,y,z in zip(numbers, typo_streets, street_suffs)]
test_data = [list(x) for x in zip(typo_full_streets, zips)]

# Initialize a graph session and declare the placeholders
sess = tf.Session()
test_address = tf.sparse_placeholder(dtype=tf.string)
test_zip = tf.placeholder(shape=[None, 1], dtype=tf.float32)
ref_address = tf.sparse_placeholder(dtype=tf.string)
ref_zip = tf.placeholder(shape=[None, n], dtype=tf.float32)

# Declare the distances
zip_dist = tf.square(tf.subtract(ref_zip, test_zip))
address_dist = tf.edit_distance(test_address, ref_address, normalize=True)

# Convert the zip distance and the address distance into similarities.
zip_max = tf.gather(tf.squeeze(zip_dist), tf.argmax(zip_dist, 1))
zip_min = tf.gather(tf.squeeze(zip_dist), tf.argmin(zip_dist, 1))
zip_sim = tf.div(tf.subtract(zip_max, zip_dist), tf.subtract(zip_max, zip_min))
address_sim = tf.subtract(1., address_dist)

# Combining the two similarity function
address_weight = 0.5
zip_weight = 1. - address_weight
weighted_sim = tf.add(tf.transpose(tf.multiply(address_weight, address_sim)), tf.multiply(zip_weight, zip_sim))
top_match_index = tf.argmax(weighted_sim, 1)

# In order to use the edit distance in TensorFlow, we need to convert the address strings to a sparse vector.
def sparse_from_word_vec(word_vec):
    num_words = len(word_vec)
    indices = [[xi, 0, yi] for xi,x in enumerate(word_vec) for yi,y in enumerate(x)]
    chars = list(''.join(word_vec))
    return (tf.SparseTensorValue(indices, chars, [num_words, 1, 1]))

# Separate the address and zip codes in the reference dataset, so we can feed them into the placeholders when we loop through the test set
reference_addresses = [x[0] for x in reference_data]
reference_zips = np.array([[x[1] for x in reference_data]])

# Create the sparse tensor set of reference addresses using the function sparse_from_word_vec
sparse_ref_set = sparse_from_word_vec(reference_addresses)

# Loop through each entry of test set
for i in range(n):
    test_address_entry = test_data[i][0]
    test_zip_entry = [[test_data[i][1]]]

    # Create address sparse vectors
    test_address_repeated = [test_address_entry] * n
    sparse_test_set = sparse_from_word_vec(test_address_repeated)

    feeddict = {test_address: sparse_test_set,
                test_zip: test_zip_entry,
                ref_address: sparse_ref_set,
                ref_zip: reference_zips}
    best_match = sess.run(top_match_index, feed_dict=feeddict)
    best_street = np.array(reference_addresses)[best_match]
    [best_zips] = reference_zips[0][best_match]
    [[test_zip_]] = test_zip_entry

    print('Address:' + str(test_address_entry) + ' , ' + str(test_zip_))
    print('Match: ' + str(best_street) + ' , ' + str(best_zips))
