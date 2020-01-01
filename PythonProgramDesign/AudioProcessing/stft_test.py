#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: stft_test.py
@Time: 2019/11/16 下午4:32
@Overview:
"""
from scipy import signal
import numpy as np
from scipy.io import wavfile
import torch

filename = '../LocalData/00001.wav'
fs, audio = wavfile.read(filename)

def preemphasis(signal,coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

audio_pre = preemphasis(audio, 0.97)

f, t, Zxx = signal.stft(audio_pre,
                        fs,
                        window=signal.hamming(int(fs*0.025)),
                        noverlap=fs * 0.015,
                        nperseg=fs * 0.025,
                        nfft=512)

afft = np.fft.rfft(audio, 512)
energy = 1.0/512 * np.square(np.absolute(Zxx))
energy = np.sum(energy, 1)

from python_speech_features import fbank
fb, ener = fbank(audio, samplerate=fs, winfunc=np.hamming)
log_ener = np.log(ener)








