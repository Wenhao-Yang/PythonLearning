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
if os.path.isfile(save_file_name) & os.path.getsize(save_file_name)>0 :
    with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
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
        writer.writerow(text_data)

texts = [x[1] for x in  text_data]
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
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size=sentence_size, min_frequency=min_word_freq)
vocab_processor.fit_transform(texts)
embedding_size = len(vocab_processor.vocabulary_)

# Split the data into train and test dataset
train_indcies = np.random.choice(len(texts), round(len(texts)*0.8), replace=False)



