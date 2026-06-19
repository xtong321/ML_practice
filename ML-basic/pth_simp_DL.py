"""
use pytorch to implement DL, including Tensor, Forward pass, Loss, and Backward pass
Process:
- step-1: build network architecture
- step-2: define loss function
- step-3: select optimizer
- step-4: train loop
"""

import torch
import torch.nn as nn
import torch.optim as optim

# 1. prepare data
# assume there are 100 samples, each sample 3 features, predict 1 target value
X = torch.randn(100, 3)
y = torch.randn(100, 1)

# 2. define neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # define: input layers 3 -> hidden layers 5 -> output layers 1
        self.hidden = nn.Linear(3, 5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(5, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return X

model = SimpleNet()

# 3. define loss function and optimizer
criterion = nn.MSELoss() # mean square error
optimizer = optim.SGD(model.parameters(), lr=0.01) # SGD

# 4. train a model
epochs = 100
for epoch in range(epochs):
    # forward pass, compute predict
    outputs = model(X)

    # compute loss
    loss = criterion(outputs, y)

    # backward and update weights
    optimizer.zero_grad() # clean previous gradients
    loss.backward()       # back-prop, calculate gradients
    optimizer.step()      # update weight parameters

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


