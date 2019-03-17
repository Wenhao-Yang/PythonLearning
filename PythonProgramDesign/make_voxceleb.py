#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: make_voxceleb.py
@Time: 2019/3/16 下午2:48
@Overview:
data dir structure:
    /data/voxceleb/voxceleb1_wav/vox1_test_wav/wav/id10309/vobW27_-JyQ/00015.wav
produce files:
    spk2utt: spkid filepath
    utt2spk: produced by *.sh script
    wav.scp: uttid filepath
"""
import os
import csv

def prep_id_idname(meta_path):
    id_idname = {}
    with open(meta_path) as f:
        meta_file = csv.reader(f)
        for row in meta_file:
            if meta_file.line_num > 1:
                (id,idname,gender,country,set) = row[0].split('\t')
                id_idname[id] = idname
    return id_idname

def prep_u2s_ws(flistpath, id_idname, out_dir):
    uid2scp = []
    uid2idname = []
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(flistpath) as f:
        for line in f.readlines():
            # id11251/s4R4hvqrhFw/00006.wav
            id = line[-30:-23]
            rec_id = line[-22:-11]
            wav_name = line[-9:-1]
            uid = str(id) + '-' +str(rec_id) + '-' + str(wav_name)
            uid2scp.append((uid, line))
            uid2idname.append((uid, id_idname[id]))

    with open(out_dir, 'wav.scp','w') as f:
        for e in uid2scp:
            f.writelines(str(e[0]) + ' ' + str(e[1]))

    with open(out_dir, 'utt2spk','w') as f:
        for e in uid2idname:
            f.writelines(str(e[0]) + ' ' + str(e[1]) + '\n')

train_set_path = '/data/voxceleb/voxceleb1_wav/vox1_dev_wav/'
test_set_path = '/data/voxceleb/voxceleb1_wav/vox1_test_wav/'

train_flist_path = os.path.join(train_set_path, 'wav.flist')
test_flist_path = os.path.join(test_set_path, 'wav.flist')

id_idname_set = prep_id_idname('data/vox1_meta.csv')
prep_u2s_ws(train_flist_path, id_idname_set, 'data/voxceleb1_train')
prep_u2s_ws(test_flist_path, id_idname_set, 'data/voxceleb1_test')