import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os.path

class DataSet:
    index = 0
    count = 0
    epochs = 0
    inputs = None
    labels = None

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.count = len(self.inputs)

    def next_batch(self, batch_size):
        start = self.index
        end = self.index + batch_size

        # We are past the end of the data set so we permutate it and begin anew
        if (self.count < end):
            self.epochs += 1
            start = 0
            end = batch_size
            permutation = np.random.permutation(self.count)
            self.inputs = self.inputs[permutation]
            self.labels = self.labels[permutation]
        
        self.index = end
        
        return (self.inputs[start:end], self.labels[start:end])

        

def train_neural_net(train, test=None):
    x = tf.placeholder(tf.float32, [None, 16])
    W = tf.Variable(tf.random_normal([16, 13], stddev=0.35))
    b = tf.Variable(tf.constant(0.1, shape=[13]))
    l1 = tf.matmul(x, W) + b
    l1 = tf.nn.relu(l1)
    W2 = tf.Variable(tf.random_normal([13, 10], stddev=0.35))
    b2 = tf.Variable(tf.constant(0.1, shape=[10]))
    y = tf.matmul(l1, W2) + b2

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver({'W': W, 'b' : b, 'W2' : W2, 'b2' : b2})

    if os.path.isfile("./model.cpkt.meta"):
        print("Loading previous model.")
        saver.restore(sess, "./model.cpkt")
    else:
        print("Initializing new model.")
        sess.run(tf.global_variables_initializer())

    for i in tqdm(xrange(50000)):
        batch_xs, batch_ys = train.next_batch(1000)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    saver.save(sess, "./model.cpkt")

    if test != None:
        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: test.inputs,
                                            y_: test.labels}))

    return (x, y)

def classify((x, network), input):
    return network.eval(feed_dict={x : input})
