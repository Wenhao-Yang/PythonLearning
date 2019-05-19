#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Torch_1.6_nnModule.py
@Time: 2019/5/19 下午9:38
@Overview: The nn package defines a set of molules, which are roughly equicalent to nerual network layers. A module receives input tensors and computes output tensors, but may also hold internal state such as tensors containing learnable parameters.
"""
import torch
# N is the batch size, and N_in is input dimension
# H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package the define  the model as a sequence of layers. Each linear module computes output from input using a linear function, and holds internal Tensors for its weight and bias
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this case we will use mse
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
for t in range(500):
    # Forward Pass: Compute predicted y
    y_pred = model(x)

    # Compute and print loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Use autograd to compute the backward
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
