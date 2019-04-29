#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 9.1_ImplementRNNforSpamPrediction.py
@Time: 2019/4/16 下午7:04
@Overview: In this recipe, a standard RNN is implemented in TensorFlow to predict whether or not a text message is a spam or ham. We will use the SMS spam-collection dataset from the ML repository at UCI. The architecture we will use for prediction will be an input RNN sequence from the embedded text, and we will take the last RNN output as a prediction of spam or ham(0 or 1)
"""
import os
import re
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile

sess = tf.Session()
epochs = 20
batch_size = 250
max_sequence_length = 25
rnn_size = 10
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005
droupout_keep_prob = tf.placeholder(tf.float32)

# Download the data
