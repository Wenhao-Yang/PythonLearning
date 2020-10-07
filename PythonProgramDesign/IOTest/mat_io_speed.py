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
import profile

import kaldi_io
import lmdb
import numpy as np
import scipy.io as sio
import os
import time
import random
import matplotlib.pyplot as plt
import pickle
import h5py
import kaldiio
from kaldiio import WriteHelper
from functools import wraps


def getdirsize(dir):
    if os.path.isfile(dir):
        return os.path.getsize(dir) / 1024 / 1024.

    elif os.path.isdir(dir):
        size = 0
        for root, dirs, files in os.walk(dir):
          size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
        return size / 1024 / 1024.

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        print('Start %s ...' % function.__name__)
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        t = float(t1-t0)
        size = getdirsize(result)

        print("Total time running %s: %.4f seconds, %.4f MB.\n" %
              (function.__name__, float(t), size))
        return t, size

    return function_timer

@profile
@fn_timer
def npz_save(npz_file, kwds):
    np.savez_compressed(npz_file, **kwds)
    test_np = np.load(npz_file)

    for u in utts:
        test_np[u].shape

    return npz_file

@profile
@fn_timer
def npy_save(np_dir, kwds):
    uid_npath = {}
    for u in kwds.keys():
        np.save(os.path.join(np_dir, u + '.npy'), kwds[u])
        uid_npath[u] = os.path.join(np_dir, u + '.npy')

    for u in utts:
        test_np = np.load(uid_npath[u])
        test_np.shape

    return np_dir

@profile
@fn_timer
def mat_save(mat_file, kwds):
    sio.savemat(mat_file, kwds, do_compression=True)
    test_mat = sio.loadmat(mat_file)
    for u in utts:
        t = test_mat[u]
        t.shape

    return mat_file

@profile
@fn_timer
def kalid_io_save(ark_dir, kwds):
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

    return ark_dir

@profile
@fn_timer
def kaldiio_save(ark_dir, kwds):
    ark_file = os.path.join(ark_dir, 'test.ark')
    feat_scp = os.path.join(ark_dir, 'feat.scp')

    with WriteHelper('ark,scp:%s,%s' % (ark_file, feat_scp), compression_method=1) as writer:
        for u in kwds.keys():
            writer(str(u), kwds[u])

    d = kaldiio.load_scp(feat_scp)
    for u in kwds.keys():
        numpy_array = d[u]
        numpy_array.shape
    return ark_dir

@profile
@fn_timer
def pickle_save(pick_dir, kwds):
    pick_file = os.path.join(pick_dir, 'test.pickle')
    with open(pick_file, 'wb') as pic_f:
        pickle.dump(kwds, pic_f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(pick_file, 'rb') as pic_f:
        new_dic = pickle.load(pic_f)
    for u in utts:
        test_mat = new_dic[u]
        test_mat.shape

    return pick_dir

@profile
@fn_timer
def lmdb_save(lmdir_dir, kwds):
    lmdb_file = os.path.join(lmdir_dir, 'test.lmdb')
    data_size_per_exa = np.random.rand(234, 56).astype(np.float32).nbytes
    # print('data size per examples is: ', data_size_per_exa)
    data_size = data_size_per_exa * len(utts)

    env = lmdb.open(lmdb_file, map_size=data_size * 10)
    # map_size：
    # Maximum size database may grow to; used to size the memory mapping. If database grows larger
    # than map_size, an exception will be raised and the user must close and reopen Environment.
    # write data to lmdb
    txn = env.begin(write=True)
    # resolutions = []
    # tqdm_iter = tqdm(enumerate(utts), total=len(utts), leave=False)
    for idx, key in enumerate(utts):
        # tqdm_iter.set_description('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = kwds[key]

        # H, W = data.shape
        # resolutions.append('{:d}_{:d}'.format(H, W))
        txn.put(key_byte, data)

        if (idx + 1) % 200 == 0:
            txn.commit()
            # commit 之后需要再次 begin
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    # print('Finish writing lmdb.')

    env = lmdb.open(lmdb_file, map_size=data_size * 10)
    for key in kwds.keys():
        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        utt = np.frombuffer(buf, dtype=np.float32)

        utt.shape

    return lmdir_dir

@profile
@fn_timer
def h5py_save(h5py_dir, kwds):
    h5py_file = os.path.join(h5py_dir, 'test.h5py')

    with h5py.File(h5py_file, 'w') as f:  # 写入的时候是‘w’
        for u in kwds.keys():
            # np.save(os.path.join(np_dir, u + '.npy'), kwds[u])
            f.create_dataset(u, data=kwds[u], compression="gzip", compression_opts=5)

    with h5py.File(h5py_file, 'r') as f:  # 写入的时候是‘w’
        for u in kwds.keys():
            # np.save(os.path.join(np_dir, u + '.npy'), kwds[u])
            utt = f.get(u)[:]
            utt.shape

    return h5py_dir


if __name__ == '__main__':

    kwds = {}
    utts = []
    for i in range(10000):
        kwds['mat%s'%str(i)]=np.random.rand(234, 56).astype(np.float32)
        utts.append('mat%s'%str(i))

    random.shuffle(utts)

    npz_file = 'PythonProgramDesign/misc/test.npz'
    t1,npzsize = npz_save(npz_file, kwds)

    np_dir = 'PythonProgramDesign/misc/test_npy'
    if not os.path.exists(np_dir):
        os.makedirs(np_dir)
    t2,npysize = npy_save(np_dir, kwds)

    mat_file = 'PythonProgramDesign/misc/test.mat'
    t3,matsize = mat_save(mat_file, kwds)

    # Kaldi_io
    ark_dir = 'PythonProgramDesign/misc/kaldi_io'
    if not os.path.exists(ark_dir):
        os.makedirs(ark_dir)
    t4,kaldisize = kalid_io_save(ark_dir, kwds)

    # Kaldiio
    ark_dir = 'PythonProgramDesign/misc/kaldiio'
    if not os.path.exists(ark_dir):
        os.makedirs(ark_dir)
    t5, kaldiiosize = kaldiio_save(ark_dir, kwds)

    pick_dir = 'PythonProgramDesign/misc/pick'
    if not os.path.exists(pick_dir):
        os.makedirs(pick_dir)
    t6,picksize = pickle_save(pick_dir, kwds)

    lmdir_dir = 'PythonProgramDesign/misc/lmdir'
    if not os.path.exists(lmdir_dir):
        os.makedirs(lmdir_dir)
    t7,lmdbsize = lmdb_save(lmdir_dir, kwds)

    h5py_dir = 'PythonProgramDesign/misc/h5py'
    if not os.path.exists(h5py_dir):
        os.makedirs(h5py_dir)
    t8,h5pysize = h5py_save(h5py_dir, kwds)


    plt.figure(figsize=(12, 8))
    plt.title('Data IO')
    t = np.array([t1, t2, t3, t4, t5, t6, t7, t8])
    t = t #/min(t)
    s = np.array([npzsize, npysize, matsize, kaldisize, kaldiiosize, picksize, lmdbsize, h5pysize])
    s = s #/min(s)
    annote = ['npz', 'npys', 'mat', 'kaldi_io', 'kaldiio', 'pickle', 'lmdb', 'h5py']
    for i in range(len(t)):
        plt.scatter(t[i], s[i])
        plt.text(t[i], s[i], annote[i])

    plt.xlabel('Time (s)')
    plt.ylabel('Size (MB)')
    plt.savefig("mat.io.png")
    # plt.show()



