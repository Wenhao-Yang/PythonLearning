#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 7.5_MakingPredictionsWithWord2vec.py
@Time: 2019/3/11 11:05
@Overview: In the recipe, the prior-trained embeddings to perform sentiment analysis by training a logistic linear model to predict a good or bad movie review.
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

