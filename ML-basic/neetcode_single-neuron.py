"""
https://neetcode.io/problems/single-neuron/question

Every neural network is built from the same fundamental unit: 
a single neuron. It takes inputs, multiplies each by a weight, 
adds a bias, and passes the result through an activation function. 
Implement the forward pass of a single neuron.

A neuron takes a vector of inputs x, multiplies each by a corresponding weight 
w, adds a bias b, and passes the result through an activation function:

output = activation(∑i wi ⋅ xi + b)

Input:
x = [1.0, 2.0]
w = [0.5, 0.5]
b = 0.0
activation = sigmoid
Output: 0.81757
"""

import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        #pass

        z = np.dot(x, w) + b
        res = 0
        if activation == "sigmoid":
            res = 1 / (1+np.exp(-z))
        else: # ReLU
            res = np.max(0, z)

        return res


if __name__ == "__main__":
    x = [1.0, 2.0]
    w = [0.5, 0.5]
    b = 0.0
    activation = "sigmoid"

    Sol = Solution()
    res = Sol.forward(x, w, b, activation)
    res = np.round(res, 5)
    print("res = ", res)