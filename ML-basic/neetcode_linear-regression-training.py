"""
https://neetcode.io/problems/linear-regression-training/question
You've built the forward pass. Now make the model actually learn. 
The training loop ties together everything from this series: forward pass, 
loss computation, and gradient descent.

At each iteration, weights are updated by subtracting the gradient scaled by the learning rate:

W ← W − α⋅∇WL
where ∇WL is the gradient of the loss with respect to each weight (provided via get_derivative()), and 
α = 0.01 is already set as self.learning_rate.

Example 1:
Input:
X = [[1, 2, 3], [1, 1, 1]]
Y = [6, 3]
num_iterations = 10
initial_weights = [0.2, 0.1, 0.6]

Output:
[0.50678, 0.59057, 1.27435]
"""

import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        #pass
        N = X.shape[0]
        weights = initial_weights
        for i in range(num_iterations):
            pred = self.get_model_prediction(X, weights) # compute pred once
            # update each weight
            for j in range(len(weights)):
                gradient = self.get_derivative(pred, Y, len(X), X, j)
                weights[j] -= self.learning_rate*gradient

        return np.round(weights, 5)


if __name__ == "__main__":
    X = np.array([[1, 2, 3], [1, 1, 1]])
    Y = np.array([6, 3])
    print("X size: ", len(X), X.shape[0])

    num_iterations = 10
    initial_weights = np.array([0.2, 0.1, 0.6])
    #Output: [0.50678, 0.59057, 1.27435]
    Sol = Solution()
    weights = Sol.train_model(X, Y, num_iterations, initial_weights)
    print(weights)

