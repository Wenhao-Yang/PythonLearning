#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: TemperatureTransform.py
@Time: 2018/12/12 下午7:16
@Overview:
"""
input_t = input()
res = 0.0
sym = 'C'
if input_t[0]=='F' or input_t[0]=='f':
    res = (eval(input_t[1:]) - 32) / 1.8
    print("{}{:.2f}".format(sym, res))
elif input_t[0]=='C' or input_t[0]=='c':
    res = eval(input_t[1:]) * 1.8 + 32
    sym = 'F'
    print("{}{:.2f}".format(sym, res))


