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

def plot_cost():
    """
    Plots the cost function for binary classification with logistic regression.

    This function creates a plot illustrating the cost function for two scenarios:
    - When the true label (Y) is 1
    - When the true label (Y) is 0

    The x-axis represents the value of the logistic regression decision function (F(w,b)),
    and the y-axis represents the corresponding cost function values.

    Returns:
    None
    """
    # Generate values for F(w,b) between 0 and 1
    log_list = np.linspace(0.01, 0.99, 100)

    # Calculate the cost function for Y=1
    cost_y_1 = -np.log(log_list)

    # Plot the cost function for Y=1
    plt.plot(log_list, cost_y_1, label='Y=1')

    # Calculate the cost function for Y=0
    cost_y_0 = -np.log(1 - log_list)

    # Plot the cost function for Y=0
    plt.plot(log_list, cost_y_0, label='Y=0')

    # Set labels and legend
    plt.xlabel('F(w,b)')
    plt.ylabel('Cost Function')
    plt.legend()

# Call the function to generate and display the plot
plot_cost()
