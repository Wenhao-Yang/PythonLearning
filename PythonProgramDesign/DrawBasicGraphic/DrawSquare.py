#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: DrawSquare.py
@Time: 2018/12/12 下午9:01
@Overview:
"""
import turtle

turtle.setup(640, 480, 200, 200)
turtle.penup()
turtle.fd(-40)
turtle.pendown()

for i in range(9):
    turtle.fd(80)
    turtle.left(80)

turtle.done()