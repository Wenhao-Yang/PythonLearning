#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: weibo-visual.py
@Time: 2019/11/25 下午4:35
@Overview:
"""
import re
from urllib.parse import urlencode
import requests
from pyquery import PyQuery as pq
import time
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient

if __name__ == '__main__':
    client = MongoClient()    #连接mongodb
    db = client['weibo_1']    #建立数据库
    collection = db['weibo_6']#建立表
    tweets = []
    time = []
    name = []
    reposts = []
    attitudes = []
    comments = []
    for x in collection.find():
        tweets.append(x)
        # time.append(x['time'])
        # name.append(x['name'])
        # reposts.append(x['reposts'])
        # attitudes.append(x['attitudes'])
        # comments.append(x['comments'])
    for x in tweets:
        x['id'] = int(x['id'])
        if re.match( r'\d*小时前', x['time'], re.M|re.I):
            x['time'] = int(x['time'].rstrip('小时前')) * -60

        elif re.match(r'\d*分钟前', x['time'], re.M | re.I):
            x['time'] = int(x['time'].rstrip('分钟前')) * -1

        time.append(x['time'])
        name.append(x['name'])
        reposts.append(x['reposts'])
        attitudes.append(x['attitudes'])
        comments.append(x['comments'])
    time = np.array(time)
    time = time.reshape(time.shape[0], 1)
    comments = np.array(comments)
    comments = comments.reshape(comments.shape[0], 1)

    plot_co = np.concatenate((time, comments), axis=1)
    plot_co[:, plot_co[0].argsort()]

    times = range(len(time))
    plt.plot(plot_co[:,0], plot_co[:, 1])
    plt.show()