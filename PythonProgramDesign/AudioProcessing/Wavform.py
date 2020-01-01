#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Wavform.py
@Time: 2019/9/10 上午11:35
@Overview:
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import wave

# 读入音频。
path = "/Users/yang/PycharmProjects/PythonLearning/LocalData"
name = '00003.wav'
filename = os.path.join(path, name)

# 打开语音文件。
f = wave.open(filename, 'rb')
# 得到语音参数
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
# ---------------------------------------------------------------#
# 将字符串格式的数据转成int型
print("reading wav file......")
strData = f.readframes(nframes)
waveData = np.fromstring(strData, dtype=np.short)
# 归一化
waveData = waveData * 1.0 / max(abs(waveData))
# 将音频信号规整乘每行一路通道信号的格式，即该矩阵一行为一个通道的采样点，共nchannels行
waveData = np.reshape(waveData, [nframes, nchannels]).T  # .T 表示转置
f.close()  # 关闭文件
print("file is closed!")
# ----------------------------------------------------------------#
'''绘制语音波形'''
print("plotting signal wave...")
time = np.arange(0, nframes) * (1.0 / framerate)  # 计算时间
time = np.reshape(time, [nframes, 1]).T
plt.plot(time[0, :nframes], waveData[0, :nframes], c="b")
plt.grid(axis="x", linestyle='--')
plt.xlabel('\时间')
plt.ylabel('强度')

# set your ticks manually
# plt.xaxis.set_ticks([1.,2.,3.,4.])

# plt.title("Original wave")
plt.show()

