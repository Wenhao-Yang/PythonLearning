#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Computing.py
@Time: 2018/12/23 下午2:36
@Overview: Draw plot for Cloud Computing class
"""
import matplotlib.pyplot as plt
import numpy as np
import re

file8G = open('../LocalData/bbp.txt')
content8G = file8G.readlines()

file4G = open('../LocalData/bbp4G.txt')
content4G = file4G.readlines()

file2C = open('../LocalData/bbp2Core.txt')
content2C = file2C.readlines()

def dataPro(content):
    reslist = []
    for i in range(len(content)):
        temp = content.pop()
        if re.match('^54ea8d04df33.*', temp)!=None:
            reslist.append(temp.split())
            continue
    resArray = np.array(reslist)
    #print(re)
    cpu = resArray[:, 2]
    meu = resArray[:, 3]
    mem = resArray[:, 6]
    # bli = resArray[:, 10]
    # blo = resArray[:, 12]

    cpu_vals = []
    meu_vals = []
    mem_vals = []
    for i in range(len(cpu)):
        cpu_vals.append(float(cpu[i][:-1]))

        mem_vals.append(float(mem[i][:-1]))
        if meu[i][-3:] == 'GiB':
            meu_vals.append(float(meu[i][:-3])*1024)
        elif meu[i][-3:] == 'MiB':
            meu_vals.append(float(meu[i][:-3]))

    return cpu_vals,meu_vals,mem_vals

c4,u4,m4 = dataPro(content4G)
c8,u8,m8 = dataPro(content8G)
c2c,u2c,m2c = dataPro(content2C)

plt.figure(figsize=(10.8, 7.2))

plt.subplot(311)
plt.plot(c4, label="4G-4Core")
plt.plot(c8, label="8G-4Core")
plt.plot(c2c, label="8G-2Core")
plt.title('CPU Useage')
plt.legend(loc='upper right', prop={'size': 8})
plt.xlabel('Time(s)')
plt.ylabel('CPU Usage(%)')

plt.subplot(312)
plt.plot(u4, label="4G-4Core")
plt.plot(u8, label="8G-4Core")
plt.plot(u2c, label="2Core-MEM USAGE")
plt.title('Memory Usage MiB')
plt.legend(loc='upper right', prop={'size': 8})
plt.xlabel('Time(s)')
plt.ylabel('Memory Usage(MiB)')

plt.subplot(313)
plt.plot(m4, label="4G-4Core")
plt.plot(m8, label="8G-4Core")
plt.plot(m2c, label="2Core-2Core")
plt.title('Memory Usage Percent')
plt.legend(loc='upper right', prop={'size': 8})
plt.xlabel('Time(s)')
plt.ylabel('Memory Usage(%)')

plt.show()

