#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: sdr_radio.py
@Time: 2020/1/1 6:26 PM
@Overview:
"""
from pylab import *
from rtlsdr import RtlSdr

# Get a list of detected device serial numbers (str)
serial_numbers = RtlSdr.get_device_serial_addresses()

# Find the device index for a given serial number
device_index = RtlSdr.get_device_index_by_serial('00000001')

sdr = RtlSdr(device_index)


# Or pass the serial number directly:
sdr = RtlSdr(serial_number='00000001')
sdr = RtlSdr()

# configure device
sdr.sample_rate = 2.4e6
sdr.center_freq = 95e6
sdr.gain = 4

samples = sdr.read_samples(256*1024)
sdr.close()

# use matplotlib to estimate and plot the PSD
psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
xlabel('Frequency (MHz)')
ylabel('Relative power (dB)')

show()