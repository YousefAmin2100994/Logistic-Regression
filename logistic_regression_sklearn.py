# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 23:04:42 2024

@author: Joe Amin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample data
x = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# Create a logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(x, y)

# Make predictions on the same data
y_pred = model.predict(x)

# Print the accuracy of the model
# Note: The accuracy calculation in the original code had an issue and has been corrected.
# Using model.score(x, y_pred) would always return 1.0, which is incorrect.
# We can use model.score(x, y) to calculate accuracy on the training data.
accuracy = model.score(x, y)
print(f'Accuracy of the model is {accuracy}')
