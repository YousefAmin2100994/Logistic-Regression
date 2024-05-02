# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:49:23 2024

@author: Joe Amin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDRegressor

# Set a wider figure size
plt.figure(figsize=(12, 5))

# First subplot: Training data and model prediction
plt.subplot(1, 2, 1)

# Define training data
x_train = np.array([0, 1, 2, 3, 4, 5])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Separate positive and negative classes
pos = y_train == 1
neg = y_train == 0

# Scatter plot for positive and negative classes
plt.scatter(x_train[pos], y_train[pos], marker='x', c='r', label='y=1 (Positive)')
plt.scatter(x_train[neg], y_train[neg], marker='o', c='b', label='y=0 (Negative)')

# Create and fit the model
model = SGDRegressor()
x_train = x_train.reshape((-1, 1))
model.fit(x_train, y_train)

# Plot the model prediction
plt.plot(x_train, model.predict(x_train), label='Model Prediction', linestyle='--')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Training Data and Model Prediction')
plt.legend()

# Second subplot: Additional data and model prediction
plt.subplot(1, 2, 2)

# Additional data for testing
x = np.array([0, 1, 2, 3, 4, 5, 10])
y = np.array([0, 0, 0, 1, 1, 1, 0])

# Separate positive and negative classes
pos = y == 1
neg = y == 0

# Scatter plot for positive and negative classes
plt.scatter(x[pos], y[pos], marker='x', c='r', label='y=1 (Positive)')
plt.scatter(x[neg], y[neg], marker='o', c='b', label='y=0 (Negative)')

# Create and fit the model
model = SGDRegressor()
x = x.reshape((-1, 1))
model.fit(x, y)

# Plot the model prediction
plt.plot(x, model.predict(x), label='Model Prediction', linestyle='--')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Additional Data and Model Prediction')
plt.legend()

# Adjust layout for better presentation
plt.tight_layout()

# Display the plots
plt.show()
