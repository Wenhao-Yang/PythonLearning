#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: setup.py
@Time: 2019/12/10 上午11:19
@Overview:
"""
from distutils.core import setup

from Cython.Build import cythonize

setup(ext_modules=cythonize("adapter.pyx"))