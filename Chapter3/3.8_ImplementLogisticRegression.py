#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 3.8_ImplementLogisticRegression.py
@Time: 2018/12/20 下午3:17
@Overview: We implement logistic regression to predict the probability of  low birthweight.
"""
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import normalize
from tensorflow.python.framework import ops
import tensorflow as tf

# Load the data through the request module and specify which features we want to use.
birthweight_url = 'http://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file = requests.get(birthweight_url)
birth_data = birth_file.text.split('\r\n')[5:]
birth_header = [x for x in birth_data[0].split('') if len(x) >= 1]
birth_data = [[float(x) for x in y.split('') if len(x) >=1] for y in birth_data[1:] if len(y) >= 1]
y_vals = np.array([x[1] for x in birth_data])
x_vals = np.array([x[2:9] for x in birth_data])

# Split the dataset into test and train sets
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Logistic regression convergence works better when the features are scaled between 0 and 1.
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

