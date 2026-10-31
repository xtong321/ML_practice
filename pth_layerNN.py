"""
use pytorch to implment Layer NN
use basic matrix multiple and tensor, not using nn.Linear or nn.Conv2d
objective:
1) use torch.nn.Parameter to implement a Full Connected Linear Layer
2) use unfold in Tensor to implement 2D conv forward
ref: https://zh.d2l.ai/chapter_convolutional-neural-networks/conv-layer.html
"""

import torch
import torch.nn as nn

def custom_conv2d(X, weight, bias=None, stride=1, padding=0):
    """
    X: (N, C_in, H_in, W_in)
    weight: (C_out, C_in, k_H, k_W)
    boas: (C_out, )
    """
    N, C_in, H_in, W_in = X.shape
    C_out, _, k_H, k_W = weight.shape

    # 1. process padding
    if padding > 0:
        X_padded = torch.zeros(N, C_in, H_in + 2*padding, W_in + 2*padding, device=X.device)
        X_padded[:, :, padding:-padding, padding:-padding] = X
    else:
        X_padded = X

    # 2. calc output size
    H_out = (H_in + 2*padding - k_H) // stride + 1
    W_out = (W_in + 2*padding - k_W) // stride + 1

    # 3. init output tensor
    out = torch.zeros(N, C_out, H_out, W_out, device=X.device)

    # 4. calc siliding win
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + k_H
            w_start = j * stride
            w_end = w_start + k_W

            # rect in current win
            window = X_padded[:, :, h_start:h_end, w_start:w_end]

            # perform multiple-add operator on all batch and channles with broadcast
            # window: (N, C_in, k_H, k_W) -> (N, 1, C_in, k_H, k_W)
            # weight: (C_out, C_in, k_H, k_W) -> (1, C_out, C_in, k_H, k_W)
            out[:, :, i, j] = torch.sum(window.unsqueeze(1)*weight.unsqueeze(0), dim=(2,3,4))

        if bias is not None:
            out += bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return out
    
    # --- validate ---
    X_test = torch.randn(2, 3, 5, 5)
    W_test = torch.randn(4, 3, 3, 3)
    B_test = torch.randn(4)

    # official std output
    official_conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1, stride=1)
    official_conv.weight.data = W_test
    official_conv.bias.data = B_test
    out_official = official_conv(X_test)

    # my output
    out_custom = custom_conv2d(X_test, W_test, B_test, padding=1, stride=1)

    # compare
    print("max-absolute-error: ", torch.max(torch.abs(out_official - out_custom)).item())