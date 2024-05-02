# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:49:23 2024

@author: Joe Amin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#style
plt.style.use('ggplot')
# Sample data
x1 = np.array([0.5, 1, 1.5, 3, 2, 1])
x2 = np.array([1.5, 1, 0.5, 0.5, 2, 2.5])
y = np.array([0, 0, 0, 1, 1, 1])

# Plotting the points based on class
for i in range(len(x1)):
    if y[i] == 0:
        plt.plot(x1[i], x2[i], marker='o', color='b')
    if y[i] == 1:
        plt.plot(x1[i], x2[i], marker='x', color='r')

# Set labels for the plot
plt.xlabel("X1")
plt.ylabel("X2")

# Plotting the decision boundary
w = np.array([1, 1])
b = -3
x_values = np.arange(0, 6)
plt.plot(x_values, 3 - x_values, label='Decision Boundary', linestyle='--', color='g')

# Add legend
plt.legend()

# Display the plot
plt.show()
