"""
Activation functions are what give neural networks the ability to learn complex patterns. Without them, a neural network is just a series of linear transformations. No matter how many layers you stack, the output would still be a linear function of the input.
Your task is to implement two of the most common activation functions from scratch using NumPy:
Sigmoid: Squishes any input to a value between 0 and 1. Used for probabilities and binary classification. The formula is: 
σ(x)=1/(1+e(−x))

ReLU (Rectified Linear Unit): Returns 0 for negative inputs and the input itself for positive values. The formula is: 
ReLU(x)=max(0,x)

Both functions should work element-wise on 1D NumPy arrays.
"""

import numpy as np
from numpy.typing import NDArray

class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        #pass

        #exp_sum = np.exp(-z)
        z_clipped = np.clip(z, -500, 500)
        exp_sum = np.exp(-z_clipped)
        res = np.round(1 / (1+exp_sum), 5)
        #np.round(1 / (1 + np.exp(-z)), 5)
        return res

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        #pass
        
        #return z if z>=0 else 0
        #res = np.max(0, z) # wrong, return a single scalar
        res = np.maximum(0, z) #element-wise comparison with 0
