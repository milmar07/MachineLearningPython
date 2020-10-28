from assignment1.sigmoid import sigmoid
from assignment1.cost_function import cost_function
from assignment1.gda import gda
from assignment1.gradient_function import gradient_function
from assignment1.logistic_Newton import logistic_Newton
from assignment1.logistic_SGD import logistic_SGD
from assignment1.predict_function import predict_function
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from assignment1 import sigmoid, cost_function, gradient_function, logistic_SGD, logistic_Newton, gda, predict_function



# Set default parameters for plots
plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# Load the dataset
data = loadmat('faces.mat')
labels = np.squeeze(data['Labels'])
labels[labels == -1] = 0    # Want labels in {0, 1}
data = data['Data']

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
num_train = X_train.shape[0]
num_test = X_test.shape[0]

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# Visualize some examples from the dataset.
samples_per_class = 10
classes = [0, 1]
train_imgs = np.reshape(X_train, [-1, 24, 24], order='F')

for y, cls in enumerate(classes):
    idxs = np.flatnonzero(np.equal(y_train, cls))
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = y * samples_per_class + i + 1
        plt.subplot(len(classes), samples_per_class, plt_idx)
        plt.imshow(train_imgs[idx])
        plt.axis('off')
        plt.title(cls)
plt.show()

# Add intercept to X and normalize to range [0, 1]
X_train = np.concatenate((np.ones((num_train, 1)), X_train/255.), axis=1)
X_test = np.concatenate((np.ones((num_test, 1)), X_test/255.), axis=1)\

"""
#Testing sigmoid
z_test = np.arange(-5, 5, 0.01)
g_test = sigmoid(z_test)
plt.plot(z_test,g_test)
plt.title('Sigmoid')
plt.show()
"""
"""
# Test your cost-function
theta_0 = np.zeros(X_train.shape[1])
l_0 = cost_function(theta_0, X_train, y_train)
print('Log-likelihood with initial theta: ', l_0)
"""
"""
# Test your implementation
x_test = np.ones([2, 10])
theta_0 = np.zeros(10)
grad_0 = gradient_function(theta_0, x_test, 1.0)
print(grad_0)
"""
"""
# Test the final classifiers
methods = ['sgd', 'newton', 'gda']

for method in methods:
    print('Evaluating {}\n'.format(method))
    start = time.time()

    if method == 'sgd':
        theta, losses = logistic_SGD(X_train, y_train)
    elif method == 'newton':
        theta, losses = logistic_Newton(X_train, y_train)
    elif method == 'gda':
        theta, losses = gda(X_train, y_train)
    else:
        raise ValueError('Method not recognised!')

    exec_time = time.time()-start
    print('Total execution time: {}s'.format(exec_time))

    pred_test, accuracy_test = predict_function(theta, X_test, y_test)
    pred_train, accuracy_train = predict_function(theta, X_train, y_train)
    print('\nTest accuracy: {}'.format(accuracy_test))
    print('Training accuracy: {}\n'.format(accuracy_train))
"""






"""
Difference between SGD and Newton 
At a local minimum (or maximum) x, the derivative of the target function f vanishes: f'(x) = 0 (assuming sufficient smoothness of f).

Gradient descent tries to find such a minimum x by using information from the first derivative of f:
It simply follows the steepest descent from the current point. This is like rolling a ball down the graph of f until it 
comes to rest (while neglecting inertia).

Newton's method tries to find a point x satisfying f'(x) = 0 by approximating f' with a linear function g and then solving
for the root of that function explicitely (this is called Newton's root-finding method). 
The root of g is not necessarily the root of f', but it is under many circumstances a good guess. 
While approximating f', Newton's method makes use of f'' (the curvature of f). 
This means it has higher requirements on the smoothness of f, but it also means that (by using more information) it often converges faster.

"""







