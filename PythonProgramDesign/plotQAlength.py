#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: plotQAlength.py
@Time: 2019/5/14 15:35
@Overview: Do statistics for the answers from the MRC.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import NullFormatter

trueAnswer = {}
rightAnswer = {}
predictAnswer = {}

with open('../LocalData/all.txt', 'r', encoding="utf-8") as allt:
    trueAnswer = json.load(allt)

with open('../LocalData/true.txt', 'r', encoding="utf-8") as allt:
    rightAnswer = json.load(allt)

with open('../LocalData/full_test_output.json', 'r', encoding="utf-8") as allt:
    predictAnswer = json.load(allt)

quesId = {}
index=0
for id in trueAnswer:
    quesId[id]=index
    index+=1



rightId = []
rightLength = []
rightIndex=[]

for id in rightAnswer:
    rightId.append(id)

for id in rightAnswer:
    rightIndex.append(quesId[id])
    rightLength.append(len(rightAnswer[id][0]['text']))
trueLength = []
predictLength = []


for id in quesId:
    trueLength.append(len(trueAnswer[id][0]['text']))
    predictLength.append(len(predictAnswer[id]))

#plt.plot(trueLength, 'r', label='True Answer Length')
#plt.plot(predictLength, 'g', label='True Answer Length')
num_id = range(0, 1500)
# plt.scatter(num_id, trueLength, alpha=0.4, marker='.', label='True Answer')
# plt.scatter(num_id, predictLength, alpha=0.4, marker='.', label='Predict Answer')
# plt.scatter(rightIndex, rightLength, alpha=0.4, marker='.', label='Right Answer')
# #plt.ylim(0, 500)
# plt.title('Answer Length')
# plt.ylabel('Length')
# plt.xlabel('Question')
# plt.legend(loc='upper right')


# the random data
x = num_id
x2 = rightIndex
y = trueLength
y2 = predictLength
y3 = rightLength
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
axScatter.scatter(x, y, alpha=0.3, marker='.')
axScatter.scatter(x, y2, alpha=0.3, marker='.')
axScatter.scatter(x2, y3, alpha=0.3, marker='.')

# now determine nice limits by hand:
binwidth = 0.25
xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
lim = (int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim((0, lim))
axScatter.set_ylim((0, 100))

bins = np.arange(-lim, lim + binwidth, binwidth)
#axHistx.hist(x, bins=bins)
axHisty.hist(y, bins=1500, orientation='horizontal')
axHisty2.hist(y2, color='darkorange', bins=1500, orientation='horizontal')
axHisty3.hist(y3, color='g', bins=610, orientation='horizontal')

#axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())
axHisty2.set_ylim(axScatter.get_ylim())
axHisty3.set_ylim(axScatter.get_ylim())
#axHisty3.set_ylim(axScatter.get_ylim())

plt.show()

