#!/usr/bin/env python
# encoding: utf-8


from scipy.io.wavfile import read
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import timeit
import speechpy

fs, signal = read('../WavFile/sound.wav')
sign = np.array(signal)
print('Sample Rate is {}'.format(fs))
#DFT
def DFT_slow(a):
    """Computer the Discrete Flourier Transform of 1D array"""
    x = np.asarray(a, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n/N)
    return np.dot(M, x)


xf=np.fft.fft(sign)
print(len(xf))

freqs = np.fft.fftfreq(len(xf), d=1/fs)
print(freqs.size)
phase = np.fft.fftshift(xf)

plt.plot(freqs, 2 * np.abs(xf) / len(xf), 'r--')
plt.xlabel("Frequency(Hz)")
plt.ylabel("Amplitude($m$)")
plt.title("Amplitude-Frequency curve")
plt.show()

plt.plot(freqs, phase, 'k-')
plt.xlabel("Frequency(Hz)")
plt.ylabel("Phase")
plt.title("Phase-Frequency curve")
plt.show()



