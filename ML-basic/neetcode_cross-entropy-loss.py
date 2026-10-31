"""
https://neetcode.io/problems/cross-entropy-loss/question
Softmax gives you probabilities, the model's confidence for each class. But how do you tell the model how wrong it was? That's what cross-entropy loss does. It compares the predicted distribution to the actual answer and produces a single number: high when the model is confidently wrong, low when it's right. This is the loss function behind almost every classifier and every token prediction in GPT.

The lower the loss, the better. A perfect prediction gives a loss of 0.

Example 1 (Binary):
Input:
y_true = [1, 0, 1]
y_pred = [0.9, 0.1, 0.8]
Output: 0.1446

Example 2 (Categorical):
Input:
y_true = [[1,0,0], [0,1,0]]
y_pred = [[0.9,0.05,0.05], [0.1,0.8,0.1]]
Output: 0.1643
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        #pass
        """
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        loss = -np.mean(y_true*(np.log(y_pred) + (1-y_pred)*np.log(1-y_pred)))
        return np.round(loss, 4)
        """
        
        N = np.size(y_true) # get array size
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1-epsilon) # clip
        res = 0 # return result
        for i in range(N):
            res += y_true[i]*np.log(y_pred[i]) + (1-y_true[i])*np.log(1-y_pred[i])

        res = -res / N
        return np.round(res, 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        #pass

        epsilon = 1e-7 # clip to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1)) # sum on class, then average
        return round(loss, 4)


if __name__ == "__main__":
    #Example 1 (Binary):    
    y_true = [1, 0, 1]
    y_pred = [0.9, 0.1, 0.8]
    #Output: 0.1446
    print(Solution().binary_cross_entropy(y_true, y_pred))

    #Example 2 (Categorical):
    y_true = [[1,0,0], [0,1,0]]
    y_pred = [[0.9,0.05,0.05], [0.1,0.8,0.1]]
    #Output: 0.1643
    print(Solution().categorical_cross_entropy(y_true, y_pred))