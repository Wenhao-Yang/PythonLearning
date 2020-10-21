#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: cudnn_test.py
@Time: 2020/10/21 20:58
@Overview:
"""
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np

parser = argparse.ArgumentParser(description='Test for cudnn.benchmark')
parser.add_argument('--run_num', type=int, default=100, help='number of runs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--use_gpu', dest='use_gpu', action='store_true', default=False, help='use gpu')
parser.add_argument('--use_benchmark', dest='use_benchmark', action='store_true', default=False, help='use benchmark')
parser.add_argument('--exp_name', type=str, default='cudnn_test', help='output file name')
args = parser.parse_args()

if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU: ', torch.cuda.get_device_name(0))
    if args.use_benchmark:
        torch.backends.cudnn.benchmark = True
        print('Using cudnn.benchmark.')
else:
    device = torch.device('cpu')
    print('Warning! Using CPU.')

images = torch.randn(args.batch_size, 3, 224, 224)
labels = torch.empty(args.batch_size, dtype=torch.long).random_(1000)

model = models.resnet101()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
images = images.to(device)
labels = labels.to(device)

time_list = []

model.train()
for itr in range(args.run_num):
    start = time.time()
    outputs = model(images)

    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end = time.time()
    print('iteration %d time: %.2f' % (itr, end-start))
    time_list.append(end-start)

with open(args.exp_name, 'w') as f:
    f.write('Device: ' + device.type + '\n')
    f.write('Use CUDNN Benchmark: ' + str(torch.backends.cudnn.benchmark) + '\n')
    f.write('Number of runs: ' + str(args.run_num) + '\n')
    f.write('Batch size: ' + str(args.batch_size) + '\n')
    f.write('Average time: %.2f s\n\n' % (np.mean(time_list)))

    for each in time_list:
        f.write(str(each))
        f.write('\n')