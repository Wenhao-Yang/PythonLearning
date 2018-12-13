#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: TextProBar.py
@Time: 2018/12/12 下午6:31
@Overview:Print text process bar, which refresh with set time
"""
import time
import math

scale = 50
print("执行开始".center(scale//2, '-'))
start = time.perf_counter()

for i in range(scale+1):
    a = '*' * i
    b = '.' * (scale - i)
    c = (i/scale) * 100
    dur = time.perf_counter() - start
    print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end="")
    time.sleep(0.1)

print("\n" +"执行结束".center(scale//2, '-'))

