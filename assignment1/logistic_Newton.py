from .cost_function import cost_function
from .gradient_function import gradient_function
from .sigmoid import sigmoid
import numpy as np
import time
import matplotlib as plt


def RMSE(X, Y, Beta):
    XB = np.dot(X, Beta)
    error = Y - XB
    rmse = np.sqrt(np.mean((error) ** 2))
    return rmse

def logistic_Newton(X, y, num_iter=10):
    Beta = np.zeros(X.shape[1])
    A=[] #stores the RMSE of trained data set
    B=[] #stores the RMSE of test data set
    for i in range(num_iter):
        g = gradient_function(X, y,Beta)
        L = 0.0001
        alpha = 0.00001
        # Beta = Beta - alpha * np.dot(H_inv,g) + 2*L*Beta
        Beta = Beta - alpha * g + 2*L*Beta
        print(Beta)
        RMSE_train = RMSE(X,y,Beta)
        RMSE_test = RMSE(X,y,Beta)
        A = np.append(A,RMSE_train)
        B = np.append(B,RMSE_test)

    plt.plot(A,range(i+1),c="r")
    plt.title("RMSE vs iterations")
    plt.plot(-B,range(i+1))
    plt.legend()
    plt.figure()