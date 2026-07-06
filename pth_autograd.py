"""
use pytorch to implement Tensor and Autograd
objective: 
1) understand Strides of tensor, and calculate graph building
2) understand basic flow

practice:
given a dataset y = 3x + 2 + epsilon, initialize weight w and bias b,
compute w and b with gradient descend, and draw loss curve
"""

import torch

# 1. setup simulated data
torch.manual_seed(0)
X = torch.randn(100, 1)
Y = 3.0 * X + 2.0 + 0.00001*torch.randn(100, 1)

# 2. init para (open gradient tracking)
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

learning_rate = 0.01

# 3. training loop
epochs = 200
for epoch in range(epochs):
    # forward prop: calc y_pred = x*w + b
    #y_pred = X.matmul(w) + b  # broadcast issue !!
    #y_pred = torch.mm(X, w) + b
    y_pred = X * w + b # X@w + b

    # calc MSE loss
    #loss = torch.mean((y_pred - Y) ** 2)
    loss = ((y_pred - Y)**2).mean()

    # back prop
    loss.backward()

    # update para under torch.no_grad() to avoid update op into compute-graph
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        #core: clean grad (otherwise accumulated grad)
        w.grad.zero_()
        b.grad.zero_()

    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}')
