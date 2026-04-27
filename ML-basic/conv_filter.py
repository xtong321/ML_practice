"""
Implement 2D conv filter by torch
https://github.com/soapisnotfat/pytorch-cifar10/blob/master/main.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import argparse
from models import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # to avoid multiple verison of OpemMP

# my personal implementation of 2D conv filter
def pth_conv(input, kernel, stride, padding=0):
    # input - N C H W (batch_size, Channels, H x W)
    # in_channels, out_channels, bias, stride, padding
    # kernel - out, in, kh, kw
    out_channels, in_channles, kh, kw = kernel.shape
    conv_layer = nn.Conv2d(in_channles, out_channels, kernel_size=(kh, kw), bias=False, stride=stride, padding=padding)
    conv_layer.weight = nn.Parameter(kernel)
    return conv_layer(input)

def pth_conv_test():
    input = torch.tensor([[1,2,3],[4,5,6],[7,8,9]]).unsqueeze(0).unsqueeze(0).float()
    kernel = torch.tensor([[1,0],[0,1]]).unsqueeze(0).unsqueeze(0).float()

    output = pth_conv(input, kernel, stride=1)
    print(output)
    assert torch.equal(output, torch.tensor([[[[6,8], [12,14]]]]))


if __name__ == '__main__':
    """import torch
    # Input tensors
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    # Operation
    z = x * y

    # Gradients
    z.backward()  # Triggers gradient computation for z with respect to x and y

    # Accessing gradients
    print(x.grad)  # Prints 3.0, which is equal to y
    print(y.grad)  # Prints 2.0, which is equal to x
    print(z.grad)  # None, as z is a scalar. Its gradient is replaced by grad_fn, the function that generated z.
    """
    #main()
    pth_conv_test()