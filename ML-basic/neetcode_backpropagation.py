"""
https://neetcode.io/problems/backpropagation/question

Backpropagation is the same idea applied to a neuron: 
figure out how much each weight contributed to the error, 
then nudge it in the direction that reduces the loss. 
It's just the chain rule, applied one link at a time from 
the output back to each weight. This is the algorithm 
that makes all of deep learning possible.

Compute the gradients of the loss with respect to every weight 
and the bias for a single sigmoid neuron.

For a single neuron with sigmoid activation and MSE loss:

Forward: 
z=w⋅x+b, then y^ = σ(z)
Loss: L=1/2(y^−y)2
Gradient: ∂L/∂wi = (y^−y)⋅y^(1−y^)⋅xi
Gradient: ∂L/∂b  = (y^−y)⋅y^(1−y^)

Example 1:
Input:
x = [1.0, 2.0]
w = [0.5, 0.5]
b = 0.0
y_true = 1.0

Output: ([-0.02721, -0.05442], -0.02721)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        #pass

        z = np.dot(x, w) + b
        y_hat = 1.0 / (1.0 + np.exp(-z))

        #error = y_hat - y_true
        #sigmoid_deriv = y_hat * (1.0 - y_hat)
        #delta = error * sigmoid_deriv
        delta = (y_hat - y_true) * y_hat * (1-y_hat)

        dL_dw = np.round(delta * x, 5)
        dL_db = round(float(delta), 5)

        return (dL_dw, dL_db)
