from assignment1.sigmoid import sigmoid
import numpy as np


def cost_function(theta, X, y):

    t = X.dot(theta)
    return -np.sum(y*np.log(sigmoid(t))+(1-y)*np.log(1-sigmoid(t))) / X.shape[0]






