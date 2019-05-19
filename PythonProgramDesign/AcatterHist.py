#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: AcatterHist.py
@Time: 2019/5/14 16:22
@Overview:
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# Fixing random state for reproducibility
np.random.seed(19680801)


# the random data
x = np.random.randn(1000)
y = np.random.randn(1000)

nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.05, 0.4
bottom, height = 0.1, 0.8
#bottom_h = left + width + 0.25
left_h = left + width + 0.02
left_h2 = left + width + 0.19
left_h3 = left + width + 0.36

rect_scatter = [left, bottom, width, height]
#rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.15, height]
rect_histy2 = [left_h2, bottom, 0.15, height]
rect_histy3 = [left_h3, bottom, 0.15, height]

# start with a rectangular Figure
plt.figure(1, figsize=(10, 4))

axScatter = plt.axes(rect_scatter)
#axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)
axHisty2 = plt.axes(rect_histy2)
axHisty3 = plt.axes(rect_histy3)

# no labels
#axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)
axHisty2.yaxis.set_major_formatter(nullfmt)
axHisty3.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(x, y)

# now determine nice limits by hand:
binwidth = 0.25
xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
lim = (int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim((-lim, lim))
axScatter.set_ylim((-lim, lim))

bins = np.arange(-lim, lim + binwidth, binwidth)
#axHistx.hist(x, bins=bins)
axHisty.hist(y, bins=bins, orientation='horizontal')
axHisty2.hist(y, bins=bins, orientation='horizontal')
axHisty3.hist(y, bins=bins, orientation='horizontal')

#axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

plt.show()