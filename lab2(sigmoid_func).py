# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:49:23 2024

@author: Joe Amin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid_function(x):
    """
    Compute the sigmoid function for a given input.

    Parameters:
    - x: Input value

    Returns:
    - result: Sigmoid of the input value
    """
    return 1/(1 + np.exp(-x))

def plot_sigmoid():
    """
    Plot the sigmoid function over a range of values.

    This function generates a plot of the sigmoid function over a range of values and adds labels and title for clarity.
    """
    x_list = np.arange(-10, 10, 1)
    y_list = sigmoid_function(x_list)
    
    # Plot the sigmoid function
    plt.plot(x_list, y_list, color='b')
    
    # Set labels and title
    plt.title("Sigmoid Function")
    plt.ylabel('sigmoid(z)')
    plt.xlabel('z')

# Call the function to generate and display the plot
plot_sigmoid()
plt.show()
