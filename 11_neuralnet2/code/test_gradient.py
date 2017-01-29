"""Test gradient of Multilayer Neural Network.
"""
from __future__ import print_function
import numpy as np
from multilayer_neural_network import MultilayerNeuralNetwork


def test_mlnn_regression_gradient():
    layers = \
        [
            {
                "type": "fully_connected",
                "num_nodes": 20
            },
            {
                "type": "fully_connected",
                "num_nodes": 10
            }
        ]
    check_gradient((10,), 3, layers, "regression")


def test_mlnn_classification_gradient():
    layers = \
        [
            {
                "type": "fully_connected",
                "num_nodes": 20
            },
            {
                "type": "fully_connected",
                "num_nodes": 10
            }
        ]
    check_gradient((10,), 3, layers, "classification")


def test_cnn_gradient():
    layers = \
        [
            {
                "type": "convolutional",
                "num_feature_maps": 2,
                "kernel_shape": (3, 3),
                "strides": (2, 2)
            },
            {
                "type": "convolutional",
                "num_feature_maps": 2,
                "kernel_shape": (3, 3),
                "strides": (1, 1)
            },
            {
                "type": "convolutional",
                "num_feature_maps": 2,
                "kernel_shape": (3, 3),
                "strides": (1, 1)
            },
            {
                "type": "fully_connected",
                "num_nodes": 10
            }
        ]
    check_gradient((1, 18, 18), 2, layers, "classification")


def check_gradient(D, F, layers, training):
    np.random.seed(0)

    n_tests = 50
    eps = 1e-6
    # You could adjust the precision here
    decimal = int(np.log10(1 / eps))

    mlnn = MultilayerNeuralNetwork(D, F, layers, training=training,
                                   std_dev=0.01, verbose=True)
    mlnn.initialize_weights(np.random.RandomState(1))

    X = np.random.rand(n_tests, *D)
    # Note that the components of a row of T have to lie within [0, 1] and sum
    # up to unity, otherwise the gradient will not be correct for softmax + CE!
    T = np.random.rand(n_tests, F)
    T /= T.sum(axis=1)[:, np.newaxis]
    # Calculate numerical and analytical gradients
    ga = mlnn.gradient(X, T)
    gn = mlnn.numerical_gradient(X, T, eps=eps)

    print("Checking gradients up to %d positions after decimal point..."
          % decimal, end="")
    np.testing.assert_almost_equal(ga, gn, decimal=decimal)
    print("OK")


if __name__ == "__main__":
    #test_mlnn_regression_gradient()
    #test_mlnn_classification_gradient()
    test_cnn_gradient()
