"""
Custom Implementation BatchNorm1d and 2d
"""

import torch
import torch.nn as nn

class StandardNetwork(nn.Module):
    def __init__(self):
        super(StandardNetwork, self).__init__()

        # 1. using BatchNorm1d for Fully Connected Layers
        self.fc_block = nn.Sequential(
            # setting bias=False
            nn.Linear(788, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # 2. using BatchNorm2d for Convolutional Layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, output_channels=16, kernel_size=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

    def forward(self, x_img, x_flat):
        out_conv = self.conv_block(x_img)
        out_fc = self.fc_block(x_flat)
        return out_conv, out_fc
        

class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(CustomBatchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum

        #learnable parametrs (init gamma=1, beta=0)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        #non-learnable running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # x shape: (batch_size, num_features)
        if self.training:
            # calculate mean and variance across the batch dim (dim=0)
            mean = x.mean(dim=0)
            # pytorch's internal forward pass uses a biased variance estimator
            var = x.var(dim=0, unbiased=False)

            # update the running statistics for use during evaluation/inference
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                unbiased_var = x.var(dim=0, unbiased=True) 
                self.running_var = (1-self.momentum)*self.running_var = self.momentum*unbiased_var
        else:
            # during evaluation, use the tracked historical stat
            mean = self.running_mean
            var = self.running_var

        # standardize the input
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # scale and shift using learnable para
        return self.gamma * x_norm + self.beta