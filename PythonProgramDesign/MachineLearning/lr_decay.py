#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: lr_decay.py
@Time: 2019/11/28 下午9:15
@Overview:
"""
import numpy as np
import matplotlib.pyplot as plt

lr = 0.1
iterations = np.arange(1000)

decay = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4]
for i in range(len(decay)):
    decay_lr = lr * (1.0 / (1.0 + decay[i] * iterations))
    plt.plot(iterations, decay_lr, label='decay={}'.format(decay[i]))

plt.ylim([0, 0.1])
plt.legend(loc='best')
plt.show()