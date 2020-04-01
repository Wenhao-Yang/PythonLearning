#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: mat_io_speed.py
@Time: 2020/4/1 12:53 PM
@Overview: Testing that mat saving in .mat is faster than .npz,
but the file size of .npz is 94% of that mat
"""
import numpy as np
import scipy.io as sio
import os
import time
import random


kwds = {}
utts = []
for i in range(1500):
    kwds['mat%s'%str(i)]=np.random.rand(234, 56)
    utts.append('mat%s'%str(i))

random.shuffle(utts)

np_file = 'PythonProgramDesign/misc/test.npz'

start_time = time.time()
np.savez_compressed(np_file, **kwds)
test_np = np.load(np_file)
for u in utts:
    test_np[u].shape

end_time = time.time()
t1 = end_time-start_time
print('Time consuming: %f' % (t1))
npsize = os.path.getsize(np_file) / 1024
print('File size: %f' % npsize)
del test_np

mat_file = 'PythonProgramDesign/misc/test.mat'
start_time = time.time()
sio.savemat(mat_file, kwds)
test_mat = sio.loadmat(mat_file)
for u in utts:
    test_mat[u].shape

end_time = time.time()
t2 = end_time-start_time
print('\nTime consuming: %f' % (t2))
mat_size = os.path.getsize(mat_file) / 1024
print('File size: %f' % mat_size)

print('\nTime cost ratio: %f %%' % (t1/t2 * 100.))
print('File size ratio: %f %%' % (npsize/mat_size*100.))



