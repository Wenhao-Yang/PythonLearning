#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Torch_1.3_Autograd.py
@Time: 2019/5/19 下午8:51
@Overview: The automgrad package provides exactly antomatic function of computation of backpasses in neural networks
"""
import torch

dtype = torch.float
device = torch.device("cpu")

# N is the batch size, and N_in is input dimension
# H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward Pass: Compute predicted y
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
