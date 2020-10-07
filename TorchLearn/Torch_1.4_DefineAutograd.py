#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Torch_1.4_DefineAutograd.py
@Time: 2019/5/19 下午9:12
@Overview: Define new autograd (ReLu) in torch.
"""
import torch
class MyReLu(torch.autograd.Function):
    """
    Implement custom autograd by subclassing torch.autograd.Function and the forward and backward passes
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return a Tensor containing the output.
        :param ctx: a context object that can be used to stash information for backward computation.
        :param input:
        :return:
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Recieve a Tensor containing the gradient of the loss with respect to the output. We need to compute the fradient of the loss with respect to the input
        :param ctx:
        :param grad_output:
        :return:
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input

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
    #Apply new relu function
    relu = MyReLu.apply

    # Forward Pass: Compute predicted y
    y_pred = relu(x.mm(w1)).mm(w2)

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
