#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: args_class.py
@Time: 2019/12/6 上午11:01
@Overview:
"""
class class_test:
    id_ = 0
    name = 'test_class'

    def __init__(self):
        self.name = 'test'

    def args_test(self, **kwargs):
        try:
            print(self.id_)
            norm = kwargs['norm']
            bo_var = kwargs['bo_var']

            print(norm, bo_var)
        except:
            print('Initialize failed!')

    def for_args(self, a):
        a[0] = 3
        print(a)


a = [1, 2, 3]
t = class_test()

t.args_test(norm='Hello', bo_var=True)
print('Original a is ', a)
t.for_args(a)
print('For_args a is ', a)



