"""
https://neetcode.io/problems/linear-regression-forward/question
You now have the full math toolkit: gradient descent to optimize, 
activation functions to add nonlinearity, softmax to get probabilities, and cross-entropy to measure error. Time to put it all together. Linear regression is the "Hello World" of machine learning, and it turns out to be a single neuron without an activation function. A neural network is really just many of these stacked together.

Implement two functions:
get_model_prediction() - computes the forward pass: 
Y^ = X ⋅ W (dot product of feature matrix and weight vector)
get_error() - computes Mean Squared Error between predictions and ground truth:

Linear Algebra (Matrix Multiplication) - The forward pass is a single matrix-vector multiply, 
and understanding shapes like 
(N×d)⋅(d×1)=(N×1) is essential
Mean Squared Error - MSE is the standard way to measure how far predictions are from truth, 
and knowing why we square errors (rather than using absolute values) matters for gradient computation

"""

import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        #pass

        prediction = np.matmul(X, weights)
        return np.round(prediction, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        #pass

        error = np.mean(np.square(model_prediction - ground_truth))
        return round(error, 5)


if __name__ == "__main__":
    X = [[1,2], [3,4], [5,6]]
    w = [0.5, -0.5]
    y = [0, 1, 0]

    np_X = np.array(X)
    np_w = np.array(w)
    np_y = np.array(y)

    #print("X shape = ", np_X)
    print("X shape = ", np_X.shape)

    pred = Solution().get_model_prediction(np_X, np_w)
    print("pred = ", pred)

    error = Solution().get_error(pred, np_y)
    print("erro = ", error)
