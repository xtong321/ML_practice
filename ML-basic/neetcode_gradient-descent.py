"""
implement gradient descent with python

Input:
1) iterations - how many gradient descent steps to execute. Can be zero (meaning return the initial value unchanged). iterations >= 0.
2) learning_rate - the step-size multiplier alpha that scales each gradient update. Strictly between 0 and 1.
3) init - the starting point for optimization. can be any number (including zero)

Apply the update rule for the specified number of iterations starting from init, then return the final value of x.
Use python round() function to return the output to 5 decimal places.
"""

class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        #pass

        min_val = init

        for _ in range(iterations):
            derivative = 2 * min_val
            min_val = min_val - learning_rate * derivative

        return round(min_val, 5)