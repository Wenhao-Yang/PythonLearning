#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: weibo.py
@Time: 2019/11/25 上午10:21
@Overview:
"""
from urllib.parse import urlencode
import requests
from pyquery import PyQuery as pq
import time
from pymongo import MongoClient

base_url = 'https://m.weibo.cn/api/container/getIndex?'

headers = {
    'Host': 'm.weibo.cn',
    'Referer': 'https://m.weibo.cn/u/2830678474',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
}

# 网易致歉： 100103type%3D1%26q%3D%2523网易致歉%2523
# 2020年部分节假日安排：100103type%3D1%26q%3D%2523%E7%BD%91%E6%98%93%E8%87%B4%E6%AD%89%2523
# 武汉下雪： 231522type%3D1%26t%3D10%26q%3D%23%E6%AD%A6%E6%B1%89%E4%B8%8B%E9%9B%AA%23
# #电视剧大赏#  231522type%3D1%26t%3D10%26q%3D%23%E7%94%B5%E8%A7%86%E5%89%A7%E5%A4%A7%E8%B5%8F%23
# #澳大利亚上千只考拉死亡# containerid=100103type%3D1%26q%3D%23%E6%BE%B3%E5%A4%A7%E5%88%A9%E4%BA%9A%E4%B8%8A%E5%8D%83%E5%8F%AA%E8%80%83%E6%8B%89%E6%AD%BB%E4%BA%A1%23&page_type=searchall

# https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D1%26q%3D%232020%E5%B9%B4%E9%83%A8%E5%88%86%E8%8A%82%E5%81%87%E6%97%A5%E5%AE%89%E6%8E%92%23
def get_page(page): #得到页面的请求，params是我们要根据网页填的，就是下图中的Query String里的参数
    params = {
        'containerid': '100103type%3D1%26q%3D%23%E6%BE%B3%E5%A4%A7%E5%88%A9%E4%BA%9A%E4%B8%8A%E5%8D%83%E5%8F%AA%E8%80%83%E6%8B%89%E6%AD%BB%E4%BA%A1%23',
        'page': page,#page是就是当前处于第几页，是我们要实现翻页必须修改的内容。
        'page_type':'searchall',
        'luicode': '10000011'
    }
    url = base_url + urlencode(params)
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print(page)
            return response.json()
    except requests.ConnectionError as e:
        print('Error', e.args)

def parse_page(json):
    if json:
        items = json.get('data').get('cards')

        for item in items:
            # for item in i.get('card_group'):
            item = item.get('mblog')
            if item == None:
                continue

            weibo = {}
            weibo['id'] = item.get('id')
            weibo['text'] = pq(item.get('text')).text()
            weibo['name'] = item.get('user').get('screen_name')
            if item.get('longText') != None :#要注意微博分长文本与文本，较长的文本在文本中会显示不全，故我们要判断并抓取。
                weibo['longText'] = item.get('longText').get('longTextContent')
            else:
                weibo['longText'] =None
            print(weibo['name'])
            print(weibo['longText'])
            weibo['attitudes'] = item.get('attitudes_count')
            weibo['comments'] = item.get('comments_count')
            weibo['reposts'] = item.get('reposts_count')
            weibo['time'] = item.get('created_at')

            yield weibo

def save_to_mongo(result):
    if collection.insert(result):
        print('Saved to Mongo')

if __name__ == '__main__':
    client = MongoClient()    #连接mongodb
    db = client['weibo_1']    #建立数据库
    collection = db['weibo_6']#建立表


    for page in range(1, 300):#循环页面
        time.sleep(1)         #设置睡眠时间，防止被封号
        json = get_page(page)
        results = parse_page(json)

        for result in results:
            save_to_mongo(result)
            print(result['time'])
