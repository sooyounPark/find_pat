import numpy as np

def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

error1 = sum_squares_error(np.array(y), np.array(t))
print("Sum of Squared Errors (first y):", error1)

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
error2 = sum_squares_error(np.array(y), np.array(t))
print("Sum of Squared Errors (second y):", error2)
