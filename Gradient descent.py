# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:59:49 2024

@author: Joe Amin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set plot style to 'ggplot' for better visualization
plt.style.use('ggplot')

# Sample data
x1 = np.array([0.5, 1, 1.5, 3, 2, 1])
x2 = np.array([1.5, 1, 0.5, 0.5, 2, 2.5])
y = np.array([0, 0, 0, 1, 1, 1])

# Plot the data points with different markers and colors based on the target variable
for i in range(len(x1)):
    if y[i] == 1:
        plt.plot(x1[i], x2[i], marker='X', color='r')
    elif y[i] == 0:
        plt.plot(x1[i], x2[i], marker='o', color='b')
plt.xlabel('X1')
plt.ylabel('X2')

def compute_function(w, b, x):
    """
    Compute the sigmoid function for logistic regression.

    Parameters:
    - w: Weight vector
    - b: Bias
    - x: Input vector

    Returns:
    - Sigmoid function result
    """
    z = np.dot(w, x) + b
    return 1 / (1 + np.exp(-z))

def compute_gradient(w, b, x, y):
    """
    Compute the gradient for logistic regression.

    Parameters:
    - w: Weight vector
    - b: Bias
    - x: Input matrix
    - y: Target variable

    Returns:
    - dj_dw: Gradient with respect to weights
    - dj_db: Gradient with respect to bias
    """
    m = x.shape[0]
    dj_dw = np.zeros(len(w))  # Initialize dj_dw as an array of zeros
    dj_db = 0
    for i in range(m):
        dj_db += (compute_function(w, b, x[i]) - y[i])
        dj_dw += (compute_function(w, b, x[i]) - y[i]) * x[i]
    return dj_dw / m, dj_db / m

def compute_gradient_descent(w, x, b, y, alpha=0.1, number_of_iteration=1000):
    """
    Perform gradient descent to optimize the logistic regression parameters.

    Parameters:
    - w: Initial weight vector
    - x: Input matrix
    - b: Initial bias
    - y: Target variable
    - alpha: Learning rate
    - number_of_iteration: Number of iterations

    Returns:
    - w: Optimized weight vector
    - b: Optimized bias
    """
    for i in range(number_of_iteration):
        dj_dw, dj_db = compute_gradient(w, b, x, y)
        w -= alpha * dj_dw
        b -= dj_db * alpha
    return w, b

X_tmp = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2, 3])
b_tmp = 1
dj_db_tmp, dj_dw_tmp = compute_gradient(w_tmp, b_tmp, X_tmp, y_tmp)
print(f"dj_db: {dj_db_tmp}")
print(f"dj_dw: {dj_dw_tmp.tolist()}")

w_tmp = np.zeros(2)
b_tmp = 0.
alph = 0.1
iters = 10000

# Use the updated weights and bias from gradient descent
w_out, b_out = compute_gradient_descent(w_tmp, X_tmp, b_tmp, y_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

# Create decision boundary using the updated weights and bias
x2 = np.arange(0, 4)
x1 = (-b_out - w_out[1] * x2) / w_out[0]

plt.plot(x1, x2, label='Decision Boundary')
plt.legend()
plt.show()
