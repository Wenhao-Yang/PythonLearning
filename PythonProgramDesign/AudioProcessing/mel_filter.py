#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: mel_filter.py
@Time: 2020/1/1 9:34 AM
@Overview: 原文链接：https://blog.csdn.net/qq_39516859/article/details/80815369
"""
import numpy as np
import pylab as plt

fs = 8000 #采样率
fl = 0
fh = fs/2 #高频
bl = 1125*np.log(1+fl/700) #把 Hz 变成 Mel
bh = 1125*np.log(1+fh/700)
p = 24 #滤波器数目
NFFT=256 #FFT的点数
B = bh-bl

#将梅尔刻度等间隔划为 mel滤波器+2, 找到其中mel滤波器数目的点
y = np.linspace(0, B, p+2)
plt.scatter(range(26), y)
plt.show()
#print(y)

#把 Mel 变成 Hz
Fb = 700*(np.exp(y/1125)-1)
#print(Fb)
plt.scatter(range(26), Fb)
plt.show()

# FFT后的频率采样点数
W2 = int(NFFT / 2 + 1)
# 采样点间对应的频率间隔
df = fs/NFFT

#采样频率值
freq = []
for n in range(0, W2): #频率的采样点
    freqs = int(n*df)
    freq.append(freqs)

plt.scatter(range(W2), freq, s=5)
plt.show()

sign_x = np.ones((100, W2))
melfb_x = np.zeros((100, 24))


bank = np.zeros((24, W2))
for k in range(1, p+1):
    f1 = Fb[k-1]
    f2 = Fb[k+1]
    f0 = Fb[k]
    n1=np.floor(f1/df)
    n2=np.floor(f2/df)
    n0=np.floor(f0/df)

    for i in range(1,W2):
        # 在滤波器范围内
        if i>=n1 and i<=n0:
            # 在该滤波器中心左边时, 权重与中心点距离成反比
            bank[k-1,i]=(i-n1)/(n0-n1)
        elif i>n0 and i<=n2:
            # 在该滤波器中心左边时, 权重与中心点距离成反比
            bank[k-1,i]=(n2-i)/(n2-n0)
    # print(k)
    # print(bank[k-1,:])
    plt.plot(freq, bank[k-1,:],'r')
plt.show()


for k in range(0, 100):
    for i in range(0, 24):
        melfb_x[k][i] = np.dot(sign_x[k], bank[i])

plt.pcolormesh(melfb_x, cmap='inferno')
plt.show()
plt.plot(np.mean(melfb_x, axis=0))
plt.show()

