#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: tqdm_iter.py
@Time: 2020/7/29 15:29
@Overview:
"""
from tqdm import tqdm

class MyNumbers:
    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        if self.a <= 20:
            x = self.a
            self.a += 1
            return x
        else:
            raise StopIteration


pbar=tqdm(MyNumbers())

for id in pbar:
    print(id)