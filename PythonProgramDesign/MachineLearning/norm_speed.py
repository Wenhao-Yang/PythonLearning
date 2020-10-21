#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: norm_speed.py
@Time: 2020/10/8 16:15
@Overview:
"""
import time
from functools import wraps
import torch
import torch.nn.functional as F

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        print('Start %s ...' % function.__name__, end='')
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        t = float(t1-t0)
        print(" Running %s: %.4f seconds." %
              (function.__name__, float(t)))
        return t

    return function_timer

@fn_timer
def F_l2_norm(input, alpha=1.0):
    # alpha = log(p * ( class -2) / (1-p))
    output = F.normalize(input, dim=1) * alpha
    return output

@fn_timer
def l2_norm(input, alpha=1.0):
    # alpha = log(p * ( class -2) / (1-p))

    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output * alpha

tensors = [torch.rand(256, 256) for i in range(1000)]
ft = 0.
lt = 0.

for t in tensors:
    ft += F_l2_norm(t)
    lt += l2_norm(t)

print('F_l2_norm time: %.4f' % ft)
print('l2_norm time: %.4f' % lt)
