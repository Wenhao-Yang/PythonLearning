#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: list_wav.py
@Time: 2020/2/10 11:38 AM
@Overview:
"""
import argparse
import pathlib
import os

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", help="List the wav file in the dir")
parser.add_argument("-w", "--output", help="write the wav file name into the file")
parser.add_argument("-s", "--suffix", default=False, help="write the wav file name into the file")
args = parser.parse_args()


local_dir = os.getcwd()
data_dir = '/'.join((local_dir, args.dir))
write_file = '/'.join((local_dir, args.output))

data_dir_path = pathlib.Path(data_dir)
assert os.path.exists(str(data_dir_path))

utts = [x for x in data_dir_path.iterdir() if x.is_file() and x.suffix == '.wav']

with open(args.output, 'w') as f:
    for x in utts:
        name = x.name if args.suffix else x.name.strip(x.suffix)
        f.write(name+'\n')
