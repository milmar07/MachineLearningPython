from .cost_function import cost_function
from .gradient_function import gradient_function
import numpy as np
import time
import tensorflow as tf

def accuracy(test_data, test_target, w , b):
    error = 0
    for i in range(len(test_target)):
        predicted_target = np.round(1.0 / (1 + np.exp(np.matmul(test_data[i], w) + b)))
        if(predicted_target != test_target[i] ):
            error += 1
    return 100*error/len(test_target)

def logistic_SGD(X, y, num_iter=100000, alpha=0.01):
    Y_predicted = 1.0 / (1 + tf.exp( tf.matmul(X , num_iter) + alpha))
    # print("a", x_train[1])
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=Y_predicted))

    # eps = 1e-10
    # maximum_likelihood = Y*tf.log(Y_predicted+eps) + (1-Y)*tf.log(1-Y_predicted+eps)
    # loss = -1 * tf.reduce_sum(maximum_likelihood)
    loss_list=[]
    test_accuracy_list=[]

    # loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_predicted))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):  # run 100 epochs
            for idx in range(int(len(X.x_train)/500)):
                Input_list = {X.x_train[idx*500:(idx+1)*500], y.y_train[idx*500:(idx+1)*500]}
                _,Loss,w_value,b_value = sess.run([optimizer,loss,w,b], feed_dict=Input_list)
                loss_list. append(Loss)


        # print(w_value,b_value)
        return w_value , b_value