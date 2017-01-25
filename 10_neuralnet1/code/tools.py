"""Helper functions for artificial neural networks.
"""

import numpy as np


def linear(A):
    """Linear activation function.

    Returns the input:

    .. math::

        g(a_f) = a_f

    Parameters
    ----------
    A : array-like, shape = (N, F)
        activations

    Returns
    -------
    Y : array-like, shape = (N, F)
        outputs
    """
    return A


def linear_derivative(Y):
    """Derivative of linear activation function.

    Parameters
    ----------
    Y : array-like, shape (N, F)
        outputs (g(A))

    Returns
    -------
    gd(Y) : array-like, shape (N, F)
        derivatives (gdot(A))
    """
    return 1


def relu(A):
    """Non-saturating activation function: Rectified Linar Unit (ReLU).

    Max-with-zero nonlinearity: :math:`max(0, a)`.

    Parameters
    ----------
    A : array-like, shape (N, J)
        activations

    Returns
    -------
    Y : array-like, shape (N, J)
        outputs
    """
    return np.maximum(A, 0)


def relu_derivative(Y):
    """Derivative of ReLU activation function.

    Parameters
    ----------
    Y : array-like, shape (N, J)
        outputs (g(A))

    Returns
    -------
    gd(Y) : array-like, shape = (N, J)
        derivatives (gdot(A))
    """
    return np.sign(Y)
