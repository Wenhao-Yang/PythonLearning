#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: weibo_pc.py
@Time: 2019/11/25 下午2:46
@Overview:
"""
import requests
from bs4 import BeautifulSoup

url = 'https://s.weibo.com/weibo?q=%23%E7%BD%91%E6%98%93%E8%87%B4%E6%AD%89%23'
wb_data = requests.get(url)
soup = BeautifulSoup(wb_data.text,'lxml')
items = soup.select('div.content > p')


for i in items:
    item = BeautifulSoup(i, 'lxml')

print(soup.select('div.info > a.name'))


