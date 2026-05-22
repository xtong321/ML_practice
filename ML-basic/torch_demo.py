"""
torch demo
ref: https://jishuzhan.net/article/1915060707575529474
"""

## test torch env
# import pytroch 
import torch
# create a tensor, 5x3
x = torch.rand(5,3)
# print the tensor
print(x)

# check if MPS is available on Mac
print(f"MPS available: {torch.backends.mps.is_available()}")

# set pytroch device
device_mps = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device_mps}")

# ----- prepare training data -----
# x_data, y_data are t2o tensors, represent input and data-label
# x_data includes data [1.0, 2.0, 3.0],
# y_data includes [2.0, 4.0, 6.0], that indicate y is 2x of x
# our objective is to train a model to learn the mapping function between x and y
x_data = torch.Tensor([[1.0], [2.0], [3.0]]).to(device_mps) # transfer to gpu
y_data = torch.Tensor([[2.0], [4.0], [6.0]]).to(device_mps)

# ----- define model -----
# define a LinearModel, derived from torch.nn.Module
class MultiLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super(MultiLayerLinearModel, self).__init__()
        # add a hidden layer, input 1 feature, output 3 features
        self.hidden = torch.nn.Linear(1, 3)
        #add a output layer, input 3 features, output 1 feature
        self.output = torch.nn.Linear(3, 1)

# define forward propogation
def forward(self, x):
    # hidden layer (using ReLU as activate function)
    x = torch.relu(self.hidden(x))
    # output layer
    y_pred = self.output(x)
    return y_pred

# ----- create model instance -----
# cheate a MultiLayerLinearModel instance
model = MultiLayerLinearModel().to(device_mps)

# loss function: torch.nn.MSELoss
criterion = torch.nn.MSELoss(reduction='sum')

# SGD as optimal, torch.optim.SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ----- train model -----
for epoch in range(1000):
    # compute y_pred from x_data
    y_pred = model(x_data)
    
    # compute loss, the diff between y_pred and y_data
    loss = criterion(y_pred, y_data)
    # print current epoch and loss
    print(epoch, loss.item())
    
    # Clear gradient cache:
    # In PyTorch, gradients are computed through backpropagation.
    # When loss.backward() is called, PyTorch automatically computes the gradients of the loss function with respect to the model parameters and stores these gradients in the .grad attribute of each parameter.
    # If the gradients are not explicitly zeroed, each call to loss.backward() will add the newly computed gradients to the previous ones, resulting in incorrect gradients.
    optimizer.zero_grad()

    # back-propogation
    loss.backward()

    # update model parameters after backward
    optimizer.step()

# print model parametres
print('Hidden layer: ')
print('w = ', model.hidden.weight)
print('b = ', model.hidden.bias)
print('Output layer: ')
print('w = ', model.output.weight)
print('b = ', model.output.bias)

# test model
x_test = torch.Tensor([[4.0]]).to(device_mps)
y_test = model(x_test)
print('y_pred = ', y_test.data)

## model para and predicted result:
"""
Hidden layer:
w = Parameter containing: tensor([[0.1732],[1.4368],[-0.9307]], requires_grad=True)
b = Parameter containing: tensor([[0.0400, -0.1946, 0.5237], requires_grad=True)
Output layer;
w = Parameter containing: tensor([[0.0159, 1.3901, 0.5729], requires_grad=True)
b = Parameter containing: tensor([0.2699], requires_grad=True)
y_pred = tensor([[8.0000]])
"""