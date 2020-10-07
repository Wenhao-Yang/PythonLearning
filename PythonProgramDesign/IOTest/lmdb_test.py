#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: lmdb_test.py
@Time: 2020/8/20 11:24
@Overview:
"""
import lmdb
import os
import time
import numpy as np
import random

from tqdm import tqdm

# from PythonProgramDesign.IOTest.mat_io_speed import getdirsize

lmdir_dir = 'PythonProgramDesign/misc/lmdir'
if not os.path.exists(lmdir_dir):
    os.makedirs(lmdir_dir)


lmdb_file = os.path.join(lmdir_dir, 'test.lmdb')

kwds = {}
utts = []
for i in range(10000):
    kwds['mat%s'%str(i)]=np.random.rand(234, 56).astype(np.float32)
    utts.append('mat%s'%str(i))

random.shuffle(utts)

data_size_per_exa = np.random.rand(234, 56).astype(np.float32).nbytes
print('data size per examples is: ', data_size_per_exa)

data_size = data_size_per_exa * len(utts)

start_time = time.time()
env = lmdb.open(lmdb_file, map_size=data_size * 10)

# map_size：
# Maximum size database may grow to; used to size the memory mapping. If database grows larger
# than map_size, an exception will be raised and the user must close and reopen Environment.

# write data to lmdb

txn = env.begin(write=True)
# resolutions = []
tqdm_iter = tqdm(enumerate(kwds.keys()), total=len(utts), leave=False)
for idx, key in tqdm_iter:
    # tqdm_iter.set_description('Write {}'.format(key))

    key_byte = key.encode('ascii')
    data = kwds[key]

    H, W = data.shape
    # resolutions.append('{:d}_{:d}'.format(H, W))
    txn.put(key_byte, data)

    if (idx + 1) % 100 == 0:
        txn.commit()
        # commit 之后需要再次 begin
        txn = env.begin(write=True)
txn.commit()
env.close()
print('Finish writing lmdb.')

env = lmdb.open(lmdb_file, map_size=data_size * 10)

for key in utts:
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    utt = np.frombuffer(buf, dtype=np.float32)

    print(key, utt.shape)
    print(utt.reshape(int(utt.shape[0]/56), 56)[0])
    print(kwds[key][0])

    break
env.close()
end_time = time.time()
t2 = end_time-start_time
print('\nnpy files Time consuming: %f' % (t2))
# npysize = getdirsize(lmdir_dir) / 1024
# print('File size: %f' % npysize)



