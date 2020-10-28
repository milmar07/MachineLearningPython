from assignment1.sigmoid import sigmoid
import numpy as np


def gradient_function(theta, X, y):
    predictions = sigmoid(np.dot(X,theta))
    grad = 1/y * np.dot(X.transpose(), (predictions - y))
    return grad
