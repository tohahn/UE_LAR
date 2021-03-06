"""Test gradient of Multilayer Neural Network.

Note that the error bounds are not very strict in this unit test. You can
increase the required precision (eps) from 1e-4 to 1e-5. Learning might
still work even though the unit test fails with stricter limits, however,
failures indicate differences to our implementation. Negligible differences
might result in small errors, e.g. drawing the initial weights from another
distribution, wrong order of bias and weights, etc.
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


def check_gradient(D, F, layers, training):
    np.random.seed(0)

    n_tests = 50
    eps = 1e-5
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
    test_mlnn_regression_gradient()
