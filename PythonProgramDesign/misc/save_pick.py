#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: save_pick.py
@Time: 2020/8/13 21:32
@Overview:
"""
import pickle
import numpy as np


a = 1
# 保存
with open('data.pickle', 'wb') as f:
  pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
# 读取
with open('data.pickle', 'rb') as f:
  b = pickle.load(f)