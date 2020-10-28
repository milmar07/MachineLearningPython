from .sigmoid import sigmoid
import numpy as np


def predict_function(theta, X, y=None):
    """
    Compute predictions on X using the parameters theta. If y is provided
    computes and returns the accuracy of the classifier as well.

    """

    preds = None
    accuracy = None
    #######################################################################
    # TODO:                                                               #
    # Compute predictions on X using the parameters theta.                #
    # If y is provided compute the accuracy of the classifier as well.    #
    #                                                                     #
    #######################################################################
    
    m=X.shape[0]
    preds = np.zeros(m)
    preds = np.round(sigmoid(X.dot(theta.T)))

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return preds, accuracy