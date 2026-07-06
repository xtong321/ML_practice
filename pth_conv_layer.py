"""
using pytorch to implement conv-layer
"""
import torch
from torch import nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

# test
X = torch.tensor([[0,1,2], [3,4,5], [6,7,8]])
K = torch.tensor([[0,1], [2,3]])
corr2d(X, K)

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

## example: learn kernel through data
# use input data X and output Y to learn kernel K

# setup a kernel
conv2d = Conv2D(kernel_size = (1,2))

step = 20
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    loss = ((Y_hat - Y)**2).sum()
    loss.backward()

    # gradiend descent
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # clean grad
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i+1)%5 == 0:
        print('Step %d, loss %.3f' % (i+1, loss.item()))


"""
output:
Step 5, loss 1.844
Step 10, loss 0.206
Step 15, loss 0.023
Step 20, loss 0.003
"""
