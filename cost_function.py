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

# Define the logistic regression model
def model(w, x, b):
    return 1 / (1 + np.exp(-(np.dot(w, x) + b)))

# Define the cost function for logistic regression
def compute_cost(w, x, b, y):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        cost += (y[i] * np.log(model(w, x[i], b)) + (1 - y[i]) * (np.log(1 - model(w, x[i], b))))
    return -cost / m

# Print the computed cost for the given parameters
print(compute_cost(np.array([1, 1]), np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]), -3, y))

# Plot the decision boundary line
x1 = np.arange(0, 4)
x2 = 3 - x1
plt.plot(x1, x2, label='Decision Boundary')
plt.xlabel("X1")
plt.ylabel('X2')
plt.legend()

# Display the plot
plt.show()
