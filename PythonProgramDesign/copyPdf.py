#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: copyPdf.py
@Time: 2019/8/9 下午4:15
@Overview:Copy pdf as listed.
"""
import re
import os, shutil
import zipfile


pdf_list = open('../LocalData/pdf_list.txt')
pro_pdf = pdf_list.readlines()
pdf_pdf_dic = {}

for pdf in pro_pdf:
    item = pdf.split('\t')
    #pdfs = item[3].split('[，；]')
    pdfs = re.split('[，；]', item[3])
    if int(item[4].strip('\n'))==0:
        continue
    pdf_pdf_dic[item[1]] = pdfs

    if int(item[4].strip('\n')) != len(pdfs):
        raise ValueError

pdfs_path = '/Users/yang/Desktop/投稿期刊的论文'
target_path = '/Users/yang/Desktop/审阅分派'
pdf_name = 'NCMMSC2019_paper_{}.pdf'
os.chdir(target_path)

os.getcwd()
for key in pdf_pdf_dic.keys():
    if not os.path.isdir(key):
        os.makedirs(key)

    pro_path = os.path.join(target_path, key)
    for pdf_num in pdf_pdf_dic[key]:
        pdf_num = pdf_num.strip()
        file_name = pdf_name.format(pdf_num)
        file_path = os.path.join(pro_path, file_name)
        if not os.path.exists(file_path):
            source_path = os.path.join(pdfs_path, file_name)
            shutil.copyfile(source_path, file_path)
        print('\rCopying pdfs for {} completed.'.format(key))


def zip_file(src_dir):
    zip_name = src_dir +'.zip'
    z = zipfile.ZipFile(zip_name,'w',zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        fpath = dirpath.replace(src_dir,'')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename),fpath+filename)
            print ('==压缩成功==')
    z.close()


