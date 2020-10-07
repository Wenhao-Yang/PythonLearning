#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: weibo_sel.py
@Time: 2019/11/30 下午10:24
@Overview:
"""
# coding=utf-8

"""  
Created on 2016-04-28 
@author: xuzhiyuan

功能: 爬取新浪微博的搜索结果,支持高级搜索中对搜索时间的限定
网址：http://s.weibo.com/
实现：采取selenium测试工具，模拟微博登录，结合PhantomJS/Firefox，分析DOM节点后，采用Xpath对节点信息进行获取，实现重要信息的抓取

"""
import pdb
import time
import datetime
import re
import os
import sys
import codecs
import shutil
import urllib
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

import selenium.webdriver.support.ui as ui
from selenium.webdriver.common.action_chains import ActionChains
import xlwt

# 先调用无界面浏览器PhantomJS或Firefox
# driver = webdriver.PhantomJS()
driver = webdriver.Firefox()


# ********************************************************************************
#                            第一步: 登陆login.sina.com
#                     这是一种很好的登陆方式，有可能有输入验证码
#                          登陆之后即可以登陆方式打开网页
# ********************************************************************************

def LoginWeibo(username, password):
    try:
        # 输入用户名/密码登录
        print
        u'准备登陆Weibo.cn网站...'
        driver.get("http://login.sina.com.cn/")
        elem_user = driver.find_element_by_name("username")
        elem_user.send_keys(username)  # 用户名
        elem_pwd = driver.find_element_by_name("password")
        elem_pwd.send_keys(password)  # 密码
        elem_sub = driver.find_element_by_xpath("//input[@class='W_btn_a btn_34px']")
        elem_sub.click()  # 点击登陆 因无name属性

        try:
            # 输入验证码
            time.sleep(10)
            elem_sub.click()
        except:
            # 不用输入验证码
            pass

        # 获取Coockie 推荐资料：http://www.cnblogs.com/fnng/p/3269450.html
        print
        'Crawl in ', driver.current_url
        print
        u'输出Cookie键值对信息:'
        for cookie in driver.get_cookies():
            print
            cookie
            for key in cookie:
                print
                key, cookie[key]
        print
        u'登陆成功...'
    except Exception, e:
        print
        "Error: ", e
    finally:
        print
        u'End LoginWeibo!\n'


# ********************************************************************************
#                  第二步: 访问http://s.weibo.com/页面搜索结果
#               输入关键词、时间范围，得到所有微博信息、博主信息等
#                     考虑没有搜索结果、翻页效果的情况
# ********************************************************************************

def GetSearchContent(key):
    driver.get("http://s.weibo.com/")
    print
    '搜索热点主题：', key.decode('utf-8')

    # 输入关键词并点击搜索
    item_inp = driver.find_element(By.CSS_SELECTOR, "input:nth-child(1)")
    # item_inp = driver.find_element_by_xpath("//input[@class='W_input']")
    item_inp.send_keys(key.decode('utf-8'))
    item_inp.send_keys(Keys.ENTER)  # 采用点击回车直接搜索
    # driver.find_element(By.CSS_SELECTOR, ".wbs-feed").click()

    time.sleep(1)
    # 获取搜索词的URL，用于后期按时间查询的URL拼接
    current_url = driver.current_url
    current_url = current_url.split('&')[
        0]  # http://s.weibo.com/weibo/%25E7%258E%2589%25E6%25A0%2591%25E5%259C%25B0%25E9%259C%2587

    global start_stamp
    global page

    # 需要抓取的开始和结束日期
    start_date = datetime.datetime(2019, 11, 26, 0)
    end_date = datetime.datetime(2019, 11, 28, 19)
    delta_date = datetime.timedelta(days=1)

    # 每次抓取一天的数据
    start_stamp = start_date
    end_stamp = start_date + delta_date

    global outfile
    global sheet

    outfile = xlwt.Workbook(encoding='utf-8')
    tweets = []
    while end_stamp <= end_date:
        page = 1

        # 每一天使用一个sheet存储数据
        sheet = outfile.add_sheet(str(start_stamp.strftime("%Y-%m-%d-%H")))

        # 通过构建URL实现每一天的查询
        url = current_url + '&typeall=1&suball=1&timescope=custom:' + str(
            start_stamp.strftime("%Y-%m-%d-%H")) + ':' + str(end_stamp.strftime("%Y-%m-%d-%H")) + '&Refer=g'
        driver.get(url)

        time.sleep(1)
        handlePage(tweets)  # 处理当前页面内容

        start_stamp = end_stamp
        end_stamp = end_stamp + delta_date

    initXLS()
    for tweet in tweets:


# ********************************************************************************
#                  辅助函数，考虑页面加载完成后得到页面所需要的内容
# ********************************************************************************

# 页面加载完成后，对页面内容进行处理
def handlePage(tweets):
    while True:
        # 之前认为可能需要sleep等待页面加载，后来发现程序执行会等待页面加载完毕
        # sleep的原因是对付微博的反爬虫机制，抓取太快可能会判定为机器人，需要输入验证码
        time.sleep(2)
        # 先行判定是否有内容
        if checkContent():
            print
            "getContent"
            tweets.append(getContent())
            # 先行判定是否有下一页按钮
            if checkNext():
                # 拿到下一页按钮
                next_page_btn = driver.find_element(By.LINK_TEXT, "下一页")
                # next_page_btn = driver.find_element_by_xpath("//a[@class='page next S_txt1 S_line1']")
                next_page_btn.click()
            else:
                print
                "no Next"
                break
        else:
            print
            "handlePage -> no Content"
            break


# 判断页面加载完成后是否有内容
def checkContent():
    # 有内容的前提是有“导航条”？错！只有一页内容的也没有导航条
    # 但没有内容的前提是有“pl_noresult”
    try:
        driver.find_element_by_xpath("//div[@class='pl_noresult']")
        flag = False
    except:
        flag = True
    return flag


# 判断是否有下一页按钮
def checkNext():
    try:
        driver.find_element(By.LINK_TEXT, "下一页")
        flag = True
    except:
        flag = False
    return flag


# 在添加每一个sheet之后，初始化字段
def initXLS():
    name = ['博主昵称',
            # '博主主页',
            # '微博认证', '微博达人',
            '微博内容', '发布时间',
            # '微博地址', '微博来源',
            '收藏', '转发', '评论', '赞']

    global row
    global outfile
    global sheet

    row = 0
    for i in range(len(name)):
        sheet.write(row, i, name[i])
    row = row + 1
    outfile.save("./crawl_output_YS.xls")


# 将dic中的内容写入excel
def writeXLS(dic):
    global row
    global outfile
    global sheet

    for k in dic:
        for i in range(len(dic[k])):
            sheet.write(row, i, dic[k][i])
        row = row + 1
    outfile.save("./crawl_output_YS.xls")


# 在页面有内容的前提下，获取内容
def getContent():
    # 寻找到每一条微博的class
    nodes = driver.find_elements_by_xpath(".//div[@class='card-wrap']/div[@class='card']")
    # pdb.set_trace

    # 在运行过程中微博数==0的情况，可能是微博反爬机制，需要输入验证码
    no_result = None
    try:
        no_result = nodes[0].find_element_by_xpath(
            ".//div[@class='card-wrap']/div[@class='card card-no-result s-pt20b40']")
    except:
        pass

    if no_result != None:
        # raw_input("No Content！")
        # url = driver.current_url
        # driver.get(url)
        # getContent()
        return
    else:
        print('Current page contained {} tweets.'.format(len(nodes)))

    dic = {}

    global page
    print
    str(start_stamp.strftime("%Y-%m-%d-%H"))
    print
    u'页数:', page
    page = page + 1
    print
    u'微博数量', len(nodes)
    # pdb.set_trace()

    for i in range(len(nodes)):
        dic[i] = []
        # nodes[i].find_element_by_xpath("//div[@class='card-wrap']/div[@class='card card-no-result s-pt20b40']")
        try:
            BZNC = nodes[i].find_element_by_xpath(".//div[@class='info']/div/a[@class='name']").text
        except:
            BZNC = ''
        # pdb.set_trace()
        print
        u'博主昵称:', BZNC
        dic[i].append(BZNC)

        # try:
        #     BZZY = nodes[i].find_element_by_xpath(".//div[@class='feed_content wbcon']/a[@class='W_texta W_fb']").get_attribute("href")
        # except:
        #     BZZY = ''
        # print u'博主主页:', BZZY
        # dic[i].append(BZZY)

        # try:
        #     WBRZ = nodes[i].find_element_by_xpath(".//div[@class='feed_content wbcon']/a[@class='approve_co']").get_attribute('title')#若没有认证则不存在节点
        # except:
        #     WBRZ = ''
        # print '微博认证:', WBRZ
        # dic[i].append(WBRZ)
        #
        # try:
        #     WBDR = nodes[i].find_element_by_xpath(".//div[@class='feed_content wbcon']/a[@class='ico_club']").get_attribute('title')#若非达人则不存在节点
        # except:
        #     WBDR = ''
        # print '微博达人:', WBDR
        # dic[i].append(WBDR)

        try:
            WBNR = nodes[i].find_element_by_xpath(".//div[@class='content']/p[@class='txt']").text
        except:
            WBNR = ''
        print
        '微博内容:', WBNR
        dic[i].append(WBNR)

        try:
            FBSJ = nodes[i].find_element_by_xpath(".//div[@class='content']/p[@class='from']/a").text
        except:
            FBSJ = ''
        print
        u'发布时间:', FBSJ
        dic[i].append(FBSJ)

        # try:
        #     WBDZ = nodes[i].find_element_by_xpath(".//div[@class='feed_from W_textb']/a[@class='W_textb']").get_attribute("href")
        # except:
        #     WBDZ = ''
        # print '微博地址:', WBDZ
        # dic[i].append(WBDZ)

        # try:
        #     WBLY = nodes[i].find_element_by_xpath(".//div[@class='feed_from W_textb']/a[@rel]").text
        # except:
        #     WBLY = ''
        # print '微博来源:', WBLY
        # dic[i].append(WBLY)

        try:
            SC_TEXT = nodes[i].find_element_by_xpath(".//div[@class='card-act']/ul/li[1]").text
            if SC_TEXT == u'\u6536\u85cf':
                SC = 0
            else:
                SC = int(SC_TEXT.lstrip(u'\u6536\u85cf '))
                # SC = int(ZF_TEXT)
        except:
            ZF = 0
        print
        '收藏:', SC
        dic[i].append(str(SC))

        try:
            ZF_TEXT = nodes[i].find_element_by_xpath(".//div[@class='card-act']/ul/li[2]").text
            if ZF_TEXT == u'\u8f6c\u53d1':
                ZF = 0
            else:
                ZF = int(SC_TEXT.lstrip(u'\u8f6c\u53d1 '))
                # ZF = int(ZF_TEXT)
        except:
            ZF = 0
        print
        '转发:', ZF
        dic[i].append(str(ZF))

        try:
            PL_TEXT = nodes[i].find_element_by_xpath(".//div[@class='card-act']/ul/li[3]").text  # 可能没有em元素
            if PL_TEXT == u'\u8bc4\u8bba':
                PL = 0
            else:
                PL = int(SC_TEXT.lstrip(u'\u8bc4\u8bba '))
                # PL = int(PL_TEXT)
        except:
            PL = 0
        print
        '评论:', PL
        dic[i].append(str(PL))

        try:
            ZAN_TEXT = nodes[i].find_element_by_xpath(".//div[@class='card-act']/ul/li[4]").text  # 可为空
            if ZAN_TEXT == '':
                ZAN = 0
            else:
                ZAN = int(ZAN_TEXT)
        except:
            ZAN = 0
        print
        '赞:', ZAN
        dic[i].append(str(ZAN))

        print
        '\n'

    # 写入Excel
    return dic


# *******************************************************************************
#                                程序入口
# *******************************************************************************
if __name__ == '__main__':
    # 定义变量
    username = '17695587536'  # 输入你的用户名
    password = '960917'  # 输入你的密码

    # 操作函数
    # LoginWeibo(username, password)       #登陆微博

    # 搜索热点微博 爬取评论
    key = '#高以翔去世#'
    GetSearchContent(key)

