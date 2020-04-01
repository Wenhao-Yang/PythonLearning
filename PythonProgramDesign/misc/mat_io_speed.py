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
import kaldi_io
import numpy as np
import scipy.io as sio
import os
import time
import random
import matplotlib.pyplot as plt


def getdirsize(dir):
   size = 0
   for root, dirs, files in os.walk(dir):
      size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
   return size


kwds = {}
utts = []
for i in range(5000):
    kwds['mat%s'%str(i)]=np.random.rand(234, 56).astype(np.float32)
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
print('npz Time consuming: %f' % (t1))
npzsize = os.path.getsize(np_file) / 1024
print('File size: %f' % npzsize)
del test_np


np_dir = 'PythonProgramDesign/misc/test_npy'
if not os.path.exists(np_dir):
    os.makedirs(np_dir)

start_time = time.time()
uid_npath = {}

for u in kwds.keys():
    np.save(os.path.join(np_dir, u+'.npy'), kwds[u])
    uid_npath[u] = os.path.join(np_dir, u+'.npy')
for u in utts:
    test_np = np.load(uid_npath[u])
    test_np.shape

end_time = time.time()
t2 = end_time-start_time
print('\nnpy files Time consuming: %f' % (t2))
npysize = getdirsize(np_dir) / 1024
print('File size: %f' % npysize)
del test_np


mat_file = 'PythonProgramDesign/misc/test.mat'
start_time = time.time()
sio.savemat(mat_file, kwds, do_compression=True)
test_mat = sio.loadmat(mat_file)
for u in utts:
    t = test_mat[u]
    t.shape

end_time = time.time()
t3 = end_time-start_time
print('\nmat file Time consuming: %f' % (t3))
mat_size = os.path.getsize(mat_file) / 1024
print('File size: %f' % mat_size)

ark_dir = 'PythonProgramDesign/misc/kaldi'
if not os.path.exists(ark_dir):
    os.makedirs(ark_dir)

start_time = time.time()
ark_file = os.path.join(ark_dir, 'test.ark')
feat_scp = os.path.join(ark_dir, 'feat.scp')
uid_feat = {}
with open(ark_file, 'wb') as ark_f, open(feat_scp, 'w') as feat_f:
    for k in kwds.keys():
        kaldi_io.write_mat(ark_f, kwds[k], key='')

        offsets = str(ark_file) + ':' + str(ark_f.tell() - len(kwds[k].tobytes()) - 15)
        feat_f.write(str(k) + ' ' + offsets + '\n')
        uid_feat[k] = offsets

for u in utts:
    test_mat = kaldi_io.read_mat(uid_feat[u])
    test_mat.shape

end_time = time.time()
t4 = end_time-start_time
print('\nkaldi file Time consuming: %f' % (t4))
kaldi_size = getdirsize(ark_dir) / 1024
print('File size: %f' % kaldi_size)



t = np.array([t1, t2, t3, t4])
t = t/min(t)
s = np.array([npzsize, npysize, mat_size, kaldi_size])
s = s/min(s)
for i in range(len(t)):
    plt.scatter(t[i], s[i])

plt.legend(['npz', 'npys', 'mat', 'kaldi'])
plt.show()



