#!/usr/bin/python

import tensorflow as tf
import scipy.io
from sklearn import preprocessing
from sklearn import cross_validation
import numpy as np
from sklearn.cross_decomposition import CCA

data_dir = "/home/ubuntu/RL_final/DATA/"

# important hyperparameters
R11 = 1.
R21 = 1.
R22 = 1.
BATCH = 20.
EPOCHS = 100

class DataSet:
    def __init__(self, train, dev, test):
        self.train = train
        self.dev = dev
        self.test = test

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def build_network(input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, keep_input, keep_hidden):
    input_layer = tf.placeholder("float", [None, input_dim])
    input_layer_drop = tf.nn.dropout(input_layer, keep_input)

    W_fc1 = weight_variable([input_dim, hidden1_dim])
    b_fc1 = bias_variable([hidden1_dim])
    h_fc1 = tf.nn.relu(tf.matmul(input_layer_drop, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_hidden)

    W_fc2 = weight_variable([hidden1_dim, hidden2_dim])
    b_fc2 = bias_variable([hidden2_dim])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_hidden)

    W_fc3 = weight_variable([hidden2_dim, hidden3_dim])
    b_fc3 = bias_variable([hidden3_dim])
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
    h_fc3_drop = tf.nn.dropout(h_fc3, keep_hidden)

    W_fc4 = weight_variable([hidden3_dim, output_dim])
    b_fc4 = bias_variable([output_dim])
    output_layer = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    return input_layer, output_layer

def read_inputs():
    fileidxJW11 = scipy.io.loadmat(data_dir + "FILEIDX/fileidxJW11.mat")
    fileidxJW13 = scipy.io.loadmat(data_dir + "FILEIDX/fileidxJW13.mat")
    fileidxJW24 = scipy.io.loadmat(data_dir + "FILEIDX/fileidxJW24.mat")
    fileidxJW30 = scipy.io.loadmat(data_dir + "FILEIDX/fileidxJW30.mat")

    JW11 = scipy.io.loadmat(data_dir + "MAT/JW11[numfr1=7,numfr2=7].mat")
    JW13 = scipy.io.loadmat(data_dir + "MAT/JW13[numfr1=7,numfr2=7].mat")
    JW24 = scipy.io.loadmat(data_dir + "MAT/JW24[numfr1=7,numfr2=7].mat")
    JW30 = scipy.io.loadmat(data_dir + "MAT/JW30[numfr1=7,numfr2=7].mat")

    scaler = preprocessing.StandardScaler()
    mfcc_features = scaler.fit_transform(preprocessing.normalize(np.transpose(JW11['MFCC'])))
    articulatory_features = scaler.fit_transform(preprocessing.normalize(np.transpose(JW11['X']).astype(float)))
    phone_labels = np.transpose(JW11['P'][0])

    lb = preprocessing.LabelBinarizer()
    lb.fit(phone_labels)
    binarized_labels = lb.transform(phone_labels)

    n_samples = mfcc_features.shape[0]
    n_mfcc_features = mfcc_features.shape[1]
    n_articulatory_features = articulatory_features.shape[1]

    permutation = np.random.permutation(n_samples)
    X1 = np.asarray([mfcc_features[i] for i in permutation])
    X2 = np.asarray([articulatory_features[i] for i in permutation])
    Y = np.asarray([binarized_labels[i] for i in permutation])

    #train, dev, test = 15948, 25948, 40948 #use 25948, 40948, 50948
    train, dev, test = 25948, 40948, 50948

    X1_tr = X1[0:train, :]
    X1_dev = X1[train+1:dev, :]
    X1_test = X1[dev+1:test, :]
    X1_all = DataSet(X1_tr, X1_dev, X1_test)

    X2_tr = X2[0:train, :]
    X2_dev = X2[train+1:dev, :]
    X2_test = X2[dev+1:test, :]
    X2_all = DataSet(X2_tr, X2_dev, X2_test)

    Y_tr = Y[0:train, :]
    Y_dev = Y[train+1:dev, :]
    Y_test = Y[dev+1:test, :]
    Y_all = DataSet(Y_tr, Y_dev, Y_test)

    baseline_acoustic_tr = X1_tr[:, 118:157]
    baseline_acoustic_dev = X1_dev[:, 118:157]
    baseline_acoustic_test = X1_test[:, 118:157]
    baseline_acoustic_all = DataSet(baseline_acoustic_tr, baseline_acoustic_dev, baseline_acoustic_test)

    return X1_all, X2_all, Y_all, baseline_acoustic_all

def main():
    sess = tf.InteractiveSession()

    X1_data, X2_data, Y_data, baseline_data = read_inputs()
    
    # set up the DCCA network
    keep_input = tf.placeholder("float")
    keep_hidden = tf.placeholder("float")
    X1_in, X1_out = build_network(273, 300, 200, 100, 50, keep_input, keep_hidden)
    X2_in, X2_out = build_network(112, 200, 150, 100, 50, keep_input, keep_hidden)

    # define the DCCA cost function
    U = tf.placeholder("float", [50, 40])
    V = tf.placeholder("float", [50, 40])
    UtF = tf.matmul(tf.transpose(U), tf.transpose(X1_out))
    GtV = tf.matmul(X2_out, V)
    canon_corr = tf.mul(1./BATCH, tf.reduce_sum(tf.mul(tf.matmul(UtF, GtV), tf.constant(np.eye(40), dtype = tf.float32))))

    corr_step = tf.train.AdamOptimizer(1e-6).minimize(- canon_corr)

    sess.run(tf.initialize_all_variables())

    # train the network
    print "Training DCCA"
    for i in range(0, EPOCHS):
        for j in range(0, len(X1_data.train), int(BATCH)):
            X1_in_batch = X1_data.train[j:(j + BATCH)]
            X2_in_batch = X2_data.train[j:(j + BATCH)]

            X1_out_batch = X1_out.eval(feed_dict = {
                X1_in : X1_in_batch,
                keep_input : 1.0,
                keep_hidden : 1.0})
            X2_out_batch = X2_out.eval(feed_dict = {
                X2_in : X2_in_batch,
                keep_input : 1.0,
                keep_hidden : 1.0})

            # compute CCA on the output layers
            cca = CCA(n_components = 40)
            cca.fit(X1_out_batch, X2_out_batch)
            U_batch = cca.x_weights_
            V_batch = cca.y_weights_

            # perform gradient step
            corr_step.run(feed_dict = {
                X1_in : X1_in_batch,
                X2_in : X2_in_batch,
                U : U_batch,
                V : V_batch,
                keep_input : 0.9,
                keep_hidden : 0.8})

            # print useful info
            print "EPOCH", i, "/ COST", canon_corr.eval(feed_dict = {
                X1_in : X1_in_batch,
                X2_in : X2_in_batch,
                U : U_batch,
                V : V_batch,
                keep_input : 1.0,
                keep_hidden : 1.0})

    # train the softmax classifier
    print "Training softmax"
    W_s = weight_variable([88, 39])
    b_s = bias_variable([39])
    baseline = tf.placeholder("float", [None, 39])
    y_true = tf.placeholder("float", [None, 39])

    # define the cost
    y_pred = tf.nn.softmax(tf.matmul(tf.concat(1, [X1_out, baseline]), W_s) + b_s)
    lr_cost = - tf.reduce_sum(y_true * tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))
    lr_step = tf.train.AdamOptimizer(1e-4).minimize(lr_cost)

    # set up accuracy checking
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess.run(tf.initialize_all_variables())

    for i in range(0, EPOCHS):
        for j in range(0, len(X1_data.train), int(BATCH)):
            lr_step.run(feed_dict = {
                X1_in : X1_data.train[j:(j + BATCH)],
                y_true : Y_data.train[j:(j + BATCH)],
                baseline : baseline_data.train[j:(j + BATCH)],
                keep_input : 1.0,
                keep_hidden : 1.0})

        print i, accuracy.eval(feed_dict = {
            X1_in : X1_data.dev,
            y_true : Y_data.dev,
            baseline : baseline_data.dev,
            keep_input : 1.0,
            keep_hidden : 1.0})

    print "Test accuracy:", accuracy.eval(feed_dict = {
        X1_in : X1_data.test,
        y_true : Y_data.test,
        baseline : baseline_data.test,
        keep_input : 1.0,
        keep_hidden : 1.0})

if __name__ == "__main__":
    main()
