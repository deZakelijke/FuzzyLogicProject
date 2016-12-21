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

    if os.path.isfile("test_features.data") and os.path.isfile("labels.data"):
        print("Loading previously saved test data.")

        test_features = np.fromfile("test_features.data")
        test_labels = np.fromfile("test_labels.data")

        test_features = test_features.reshape((len(test_features) // 16, 16))
        test_labels = test_labels.reshape((len(test_labels) // 10, 10))
    else:
        print("Extracting test features.")

        mnist = input_data.read_data_sets("MNIST/", one_hot=True)
        images = np.ceil(mnist.test.images.reshape((10000, 28, 28)))
        labels = mnist.test.labels

        (test_features, test_labels) = zip(*[(f.feature_vector(), l) for (im, l) in zip(images, labels) for f in Feature.recognize_features(skeletonize(im))])

        (test_features, test_labels) = (np.array(test_features), np.array(test_labels))

        test_features.tofile("test_features.data")
        test_labels.tofile("test_labels.data")


    print("Found", len(features), "features.")

    data = DataSet(features, labels)
    test_data = DataSet(test_features, test_labels)

    print("Training neural net.")

    network = train_neural_net(data, test_data)

    print(classify(network, test_features[0]))
    print(test_labels[0])

if __name__ == "__main__":
    main()