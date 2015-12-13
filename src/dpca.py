#!/usr/bin/python

import input_data
import tensorflow as tf
import numpy as np
from scipy.linalg import svd

def weight_variable(shape):
    '''
    Create a weight variable with a normal initialization distribution
    '''

    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    Create a bias variable with a constant initialization distribution
    '''

    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def main():
    # read in the data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

    # load the interactive session
    sess = tf.InteractiveSession()

    # define architecture of the network
    x = tf.placeholder("float", [None, 784])

    W_x1 = weight_variable([784, 1000])
    W_x2 = weight_variable([1000, 750])
    W_x3 = weight_variable([750, 500])

    b_x1 = bias_variable([1000])
    b_x2 = bias_variable([750])
    b_x3 = bias_variable([500])

    h_x1 = tf.nn.relu(tf.matmul(x, W_x1) + b_x1)
    h_x2 = tf.nn.relu(tf.matmul(h_x1, W_x2) + b_x2)
    h_x3 = tf.nn.relu(tf.matmul(h_x2, W_x3) + b_x3)

    # initialize all defined Variables
    init = tf.initialize_all_variables()
    sess.run(init)

    old_variables = set(tf.all_variables())

    # train the network
    for i in range(0, 100):
        batch_x, batch_y = mnist.train.next_batch(200)

        # estimate covariance of output layer
        batch_h_x3 = h_x3.eval(feed_dict = {
            x : batch_x})
        covariance = np.cov(batch_h_x3.transpose())

        # perform pca
        U, S, Vt = svd(covariance)
        U = U.astype(np.float32)

        # define the cost as ||h_x3 - UU'h_x3||_2
        cost = tf.nn.l2_loss(h_x3 - tf.matmul(tf.matmul(h_x3, U[:,0:250]), U[:,0:250].transpose()))

        # define the training step in terms of the cost
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

        # figue out which variables to initialize
        new_variables = set(tf.all_variables())
        added_variables = new_variables - old_variables
        init = tf.initialize_variables(added_variables)
        sess.run(init)
        old_variables = new_variables

        # perform a step
        train_step.run(feed_dict = {
            x : batch_x})

        if i % 100 == 0:
            print i

    # logistic regression
    W_x1_f = tf.constant(W_x1.eval())
    W_x2_f = tf.constant(W_x2.eval())
    W_x3_f = tf.constant(W_x3.eval())

    b_x1_f = tf.constant(b_x1.eval())
    b_x2_f = tf.constant(b_x2.eval())
    b_x3_f = tf.constant(b_x3.eval())

    h_x1_s = tf.nn.relu(tf.matmul(x, W_x1_f) + b_x1_f)
    h_x2_s = tf.nn.relu(tf.matmul(h_x1_s, W_x2_f) + b_x2_f)
    h_x3_s = tf.nn.relu(tf.matmul(h_x2_s, W_x3_f) + b_x3_f)
    h_x4_s = tf.matmul(h_x3, U[:,0:250])

    W_s = weight_variable([250, 10])
    b_s = bias_variable([10])

    y_true = tf.placeholder("float", [None, 10])
    y_pred = tf.nn.softmax(tf.matmul(h_x4_s, W_s) + b_s)

    cross_entropy = - tf.reduce_sum(y_true * tf.log(y_pred))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.initialize_all_variables()
    sess.run(init)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    for i in range(0, 10000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict = {
            x: batch_x,
            y_true: batch_y})

        if i % 100 == 0:
            print "Iteration %d accuracy %f" % (i, accuracy.eval(feed_dict = {
                x : batch_x,
                y_true : batch_y}))

    print "Teste accuracy %f" % accuracy.eval(feed_dict = {
        x : mnist.test.images,
        y_true : mnist.test.labels})

if __name__ == "__main__":
    main()
