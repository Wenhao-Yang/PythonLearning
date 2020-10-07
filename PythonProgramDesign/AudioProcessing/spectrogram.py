#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: spectrogram.py
@Time: 2019/10/20 下午10:00
@Overview:
"""
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from python_speech_features import fbank

from scipy.fftpack import fft,ifft
import torch

fs, wav = wavfile.read('LocalData/00002.wav')
window = signal.hamming(512)
f, t, Sxx = signal.spectrogram(wav, fs, noverlap=10, window=window, nfft=512)
plt.pcolormesh(t, f, Sxx, cmap='inferno')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

spectrum, freqs, ts, fig = plt.specgram(wav,
                                        NFFT=512,
                                        Fs=fs,
                                        window=np.hanning(M=512),
                                        noverlap=10,
                                        mode='default',
                                        scale_by_freq=True,
                                        sides='default',
                                        scale='dB',
                                        xextent=None)  # 绘制频谱图

plt.ylabel('Frequency')
plt.xlabel('Time')
plt.title("Spectrogram")
plt.show()

# fft_wav = fft(wav, n=512, window=np.hamming(25))
# fbank(wav, fs, 0.025, 0.01, 26, 512, winfunc=np.hamming(400))

# 包络图
from scipy import interpolate

mag = np.sum(Sxx, axis=1)
x_int = np.linspace(0, 7969, 1000)
tck = interpolate.splrep(8000*np.arange(0, 257)/257, np.log(mag), k=3, s=50)
y_int = interpolate.splev(x_int, tck, der = 0)
plt.plot(x_int, y_int, linestyle='-', linewidth=0.75, color='k')
plt.ylabel('magnitude')
plt.xlabel('frequency')
plt.show()





