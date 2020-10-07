#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Voxceleb.py
@Time: 2019/5/23 下午4:44
@Overview:
"""
from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torch.utils.data import Dataset

def default_loader(wav):

    return Image.open(img)

class voxceleb(Dataset):
    def __init__(self,
                 wav_path,
                 spect_transform=None,
                 loader=default_loader):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.img_list = [
                os.path.join(img_path, i.split()[0]) for i in lines
            ]
            self.label_list = [i.split()[1] for i in lines]
        self.img_transform = img_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        # img = self.loader(img_path)
        img = img_path
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label

    def __len__(self):
        return len(self.label_list)
