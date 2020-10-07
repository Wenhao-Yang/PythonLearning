#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: stdin_test.py
@Time: 2020/3/31 12:10 PM
@Overview:
"""
import sys
import subprocess

command = 'cat LocalData/00001.wav'
p = subprocess.Popen(command, shell = True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

[stdout, stderr] = p.communicate()

print(stdout)
