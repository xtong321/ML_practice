"""
Softmax converts a vector of raw scores (called logits) into a probability distribution. 
The output values are all positive and sum to 1, making them interpretable as probabilities.

softmax(z_i) = exp(z_i - max(z)) / sum(exp(z_i - max(z)))

Example 1:
Input: z = [1, 2, 3]
Output: [0.09, 0.2447, 0.6652]
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        #pass
        z_max = np.max(z)
        exp_sum = np.sum(np.exp(z - z_max))
        res = np.round(np.exp(z-z_max) / exp_sum, 4)

        shifted = z - np.max(z)
        exps = np.exp(shifted)
        res = np.round(exps / np.sum(exps), 4)

        return res

if __name__ == "__main__":
    z = [1, 2, 3]
    print(Solution().softmax(z))
