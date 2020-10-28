from assignment1.sigmoid import sigmoid
import numpy as np


def cost_function(theta, X, y):
    m = len(y)
    predictions = sigmoid(np.dot(X,theta))
    error = (-y * np.log(predictions)) - ((1-y)*np.log(1-predictions))

    cost = 1/m * sum(error)

    return cost






