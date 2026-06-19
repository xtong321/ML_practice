"""
implement softmax with pytorch
"""
import torch

# 1. implementation
def my_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))

def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # subtract max_val to avoid overflow
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    exps = torch.exp(x - max_val)
    return exps / torch.sum(exps, dim=dim, keepdim=True)

# 2. validate the code
x = torch.randn(2, 3)

# validate sigmoid
assert torch.allcose(my_sigmoid(x), torch.sigmoid(x)), "my_sigmoid imp error!"

# validate softmax
assert torch.allclose(my_softmax(x, dim=-1), torch.softmax(x, dim=-1)), "my_softmax imp error!"

print("validation passed!")