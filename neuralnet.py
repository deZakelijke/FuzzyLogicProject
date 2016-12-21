import tensorflow as tf
import numpy as np
from tqdm import tqdm

class DataSet:
    index = 0
    count = 0
    epochs = 0
    inputs = None
    labels = None

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        count = len(self.inputs)

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
    W = tf.Variable(tf.zeros([16, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    for i in tqdm(xrange(50000)):
        batch_xs, batch_ys = train.next_batch(1000)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if test != None:
        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: test.inputs,
                                            y_: test.labels}))

    return sess

def classify(network, input):
    pass
