from __future__ import print_function
from __future__ import division
from features import Feature
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from skimage.morphology import skeletonize
import os.path

from neuralnet import *

def main():

    features = None
    labels = None

    if os.path.isfile("features.data") and os.path.isfile("labels.data"):
        print("Loading previously saved training data.")

        features = np.fromfile("features.data")
        labels = np.fromfile("labels.data")

        features = features.reshape((len(features) // 16 , 16))
        labels = labels.reshape((len(labels) // 10, 10))
    else:
        print("Extracting features.")

        mnist = input_data.read_data_sets("MNIST/", one_hot=True)
        images = np.ceil(mnist.train.images.reshape((55000, 28, 28)))
        labels = mnist.train.labels

        (features, labels) = zip(*[(f.feature_vector(), l) for (im, l) in zip(images, labels) for f in Feature.recognize_features(skeletonize(im))])

        (features, labels) = (np.array(features), np.array(labels))

        features.tofile("features.data")
        labels.tofile("labels.data")

    if os.path.isfile("test_features.data") and os.path.isfile("test_labels.data") and os.path.isfile("test_lengths.data"):
        print("Loading previously saved test data.")

        test_features = np.fromfile("test_features.data")
        test_labels = np.fromfile("test_labels.data")
        test_lengths = np.fromfile("test_lengths.data", dtype=np.int)

        test_features = test_features.reshape((len(test_features) // 16, 16))
        test_labels = test_labels.reshape((len(test_labels) // 10, 10))
    else:
        print("Extracting test features.")

        mnist = input_data.read_data_sets("MNIST/", one_hot=True)
        images = np.ceil(mnist.test.images.reshape((10000, 28, 28)))
        test_labels = mnist.test.labels

        tfs = [Feature.recognize_features(skeletonize(im)) for im in images]
        test_lengths = np.array([len(f) for f in tfs])
        test_features = [f.feature_vector() for fs in tfs for f in fs]

        test_features = np.array(test_features)

        test_features.tofile("test_features.data")
        test_lengths.tofile("test_lengths.data")
        test_labels.tofile("test_labels.data")


    print("Found", len(features), "features.")

    #(features, labels) = group(features, labels)

    data = DataSet(features, labels)

    test_data = DataSet(test_features, spread(test_labels, test_lengths))

    print("Training neural net.")

    network = train_neural_net(data, test_data)

    print("Testing with integral.")

    print(test(network, test_features, test_labels, test_lengths))

def group(features, labels):
    d = dict()

    for (f, l) in zip(features, labels):
        fs = Feature.from_vector(f)
        if (fs in d):
            d[fs] += l
        else:
            d[fs] = l

    (features, labels) = zip(*[(f.feature_vector(), l) for (f, l) in d.iteritems()])

    (features, labels) = (np.array(features), np.array(labels))

    labels = np.minimum(labels, 1)

    return (features, labels)

def test(network, features, labels, lengths):
    classes = classify(network, features)

    score = 0.0

    p = True

    for (l, c) in zip(labels, chunk(classes, lengths)):
        s = fuzzy_integral(c)

        if p:
            print(s)
            print(l)
            p = False

        if (np.argmax(s) == np.argmax(l)):
            score += 1.0

    return score / len(labels)

def spread(xs, chunk_sizes):
    ret = []

    for (x, c) in zip(xs, chunk_sizes):
        ret += [x] * c

    return np.array(ret)
    

def chunk(xs, chunk_sizes):
    s = 0

    ret = []

    for l in chunk_sizes:
        if (l > 0):
            ret.append(xs[s:l])
            s += l

    return ret
    
# scoreMatrix is a 10xN numpy.matrix
# N is the number of features in a digit
def fuzzy_integral(score_matrix):
    score_matrix = np.transpose(score_matrix)
    score_shape = np.shape(score_matrix)
    score_matrix = np.sort(score_matrix)

    for i in range(score_shape[0]):
        for j in range(score_shape[1]):
            score_matrix[i,j] = score_matrix[i,j] * np.exp(-j)

    score_vector = np.sum(score_matrix, axis=1)
    return score_vector

if __name__ == "__main__":
    main()