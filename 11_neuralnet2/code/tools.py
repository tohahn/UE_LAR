"""Helper functions for artificial neural networks.
"""

import numpy as np


def softmax(A):
    """Softmax activation function.

    The outputs will be interpreted as probabilities and thus have to
    lie within [0, 1] and must sum to unity:

    .. math::

        g(a_f) = \\frac{\\exp(a_f)}{\\sum_{f'} \\exp(a_{f'})}.

    To avoid numerical problems, we substract the maximum component of
    :math:`a` from all of its components before we calculate the output. This
    is mathematically equivalent.

    Parameters
    ----------
    a : array-like, shape (N, F)
        activations

    Returns
    -------
    y : array-like, shape (N, F)
        outputs
    """
    A = np.exp(A - np.amax(A, axis=1, keepdims=True))
    return A / np.sum(A, axis=1, keepdims=True)

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


try:
    from numba import jit
    has_numba = True
except ImportError:
    import scipy.weave
    has_numba = False


if has_numba:

    @jit
    def convolve(feature_maps, kernels, bias, stride_y=1, stride_x=1):
        """Convolve I 2-dimensional feature maps to generate J output feature maps.

        Parameters
        ----------
        feature_maps : array-like, shape (N, I, Y, X)
            input feature maps

        kernels: array-like, shape (J, I, Y_k, X_k)
            filter kernels

        bias: array-like, shape (J, I)
            bias matrix

        stride_y: int, optional
            column step size

        stride_x: int, optional
            row step size

        kwargs : dictionary, optional
            Optional arguments

        Returns
        -------
        A : array-like, shape (N, J, Y - 2 * floor(Y_k / 2) / stride_y,
                                     X - 2 * floor(X_k / 2) / stride_x)
            convolved feature maps
        """
        # feature_maps(n, i, yi, xi) - the output of the i-th input feature map
        #                              at position (xi, yi) for the n-th sample
        # kernels(j, i, yk, xk)      - convolution kernel between i-th input
        #                              feature map and j-th output feature map
        #                              at position (xk, yk)
        N = feature_maps.shape[0]     # number of samples
        I = feature_maps.shape[1]     # number of input feature maps
        Y = feature_maps.shape[2]     # number of input feature map rows
        X = feature_maps.shape[3]     # number of input feature map columns
        J = kernels.shape[0]          # number of output feature maps
        assert I == kernels.shape[1]
        Y_k = kernels.shape[2]        # number of filter kernel rows
        X_k = kernels.shape[3]        # number of filter kernel columns
        # number of output feature map rows
        Y_o = (Y - 2 * (Y_k // 2)) // stride_y
        # number of output feature map columns
        X_o = (X - 2 * (X_k // 2)) // stride_x

        # the activation of the j-th output feature map at pixel (xo, yo)
        # for the n-th sample
        A = np.zeros((N, J, Y_o, X_o))

        ############################################################################
        # Here you should compute the forward propagation for a convolutional layer.
        # You might want to remove the @jit at the beginning of the signature to
        # get helpful error messages instead of segmentation faults.
        for n in range(N):
            for j in range(J):
                for y_o in range(Y_o):
                    for x_o in range(X_o):
                        for i in range(I):
                            for y_k in range(Y_k):
                                for x_k in range(X_k):
                                    A[n,j,y_o,x_o] += kernels[j,i,y_k,x_k] * feature_maps[n,i,y_o*stride_y+y_k, x_o*stride_x+x_k]
                            A[n,j,y_o,x_o] += bias[j,i]
        ############################################################################

        return A


    @jit
    def back_convolve(feature_maps, kernels, bias, Deltas, stride_y=1, stride_x=1):
        """Convolve I 2-dimensional feature_maps to generate J output feature_maps.

        Parameters
        ----------
        feature_maps : array-like, shape = (N, I, Y, X)
            input feature maps

        kernels : array-like, shape = (J, I, Y_k, X_k)
            filter kernels

        bias : array-like, shape = (J, I)
            bias matrix

        Deltas : array-like, shape = (N, J, Y-2*floor(Y_k/2)/stride_y,
                                      X-2*floor(X_k/2)/stride_x)
            deltas

        stride_y : int, optional
            column step size

        stride_x : int, optional
            row step size

        kwargs : dictionary, optional
            Optional arguments

        Returns
        -------
        der : array-like, shape = [J, I, Y_k, X_k]
            filter kernel derivatives

        derb : array-like, shape = [J, I]
            bias derivatives

        dEdX : array-like, shape = [N, I, Y, X]
            errors
        """
        # feature_maps(n, i, yi, xi) - the output of the i-th input feature map
        #                              at position (xi, yi) for the n-th sample
        # kernels(j, i, yk, xk)      - convolution kernel between i-th input
        #                              feature map and j-th output feature map
        #                              at position (xk, yk)
        # Deltas(n, j, yo, xo)       - derivative of the error function for the
        #                              n-the sample with respect to the activation
        #                              (xo, yo) from the j-th output feature map
        N = feature_maps.shape[0]     # number of samples
        I = feature_maps.shape[1]     # number of input feature maps
        Y = feature_maps.shape[2]     # number of input feature map rows
        X = feature_maps.shape[3]     # number of input feature map columns
        J = kernels.shape[0]          # number of output feature maps
        assert I == kernels.shape[1]
        Y_k = kernels.shape[2]        # number of filter kernel rows
        X_k = kernels.shape[3]        # number of filter kernel columns
        # number of output feature map rows
        Y_o = (Y - 2 * (Y_k // 2)) // stride_y
        # number of output feature map columns
        X_o = (X - 2 * (X_k // 2)) // stride_x

        # derivative of the error function with respect to the value of the
        # filter kernel between the j-th output feature map and the i-th
        # input feature map at position (xk, yk), this is a sum over all samples
        der = np.zeros_like(kernels)
        # derivative of the error function with respect to the bias between the
        # j-th output feature map and the i-th input feature map, this is a sum
        # over all samples
        derb = np.zeros_like(bias)
        # derivative of the error function for the n-th sample with respect
        # to the pixel (xi, yi) from the i-th input feature map
        dEdX = np.zeros_like(feature_maps)

        for n in range(N):
            for j in range(J):
                for i in range(I):
                    yo = 0
                    yi = 0
                    while yo < Y_o:
                        xo = 0
                        xi = 0
                        while xo < X_o:
                            tmp = bias[j, i]

                            yk = 0
                            yik = yi
                            while yk < Y_k:
                                xk = 0
                                xik = xi
                                while xk < X_k:
                                    dEdX[n, i, yik, xik] += (kernels[j, i, yk, xk] *
                                                             Deltas[n, j, yo, xo])
                                    der[j, i, yk, xk] += (Deltas[n, j, yo, xo] *
                                                          feature_maps[n, i, yik, xik])

                                    xk += 1
                                    xik += 1
                                yk += 1
                                yik += 1

                            derb[j, i] += Deltas[n, j, yo, xo]

                            xo += 1
                            xi += stride_x

                        yo += 1
                        yi += stride_y

        return der, derb, dEdX

else: # Use scipy.weave

    def convolve(feature_maps, kernels, bias, stride_y=1, stride_x=1, **kwargs):
        """Convolve I 2-dimensional feature maps to generate J output feature maps.

        Parameters
        ----------
        feature_maps : array-like, shape (N, I, Y, X)
            input feature maps

        kernels: array-like, shape (J, I, Y_k, X_k)
            filter kernels

        bias: array-like, shape (J, I)
            bias matrix

        stride_y: int, optional
            column step size

        stride_x: int, optional
            row step size

        kwargs : dictionary, optional
            Optional arguments

        Returns
        -------
        A : array-like, shape (N, J, Y - 2 * floor(Y_k / 2) / stride_y,
                                     X - 2 * floor(X_k / 2) / stride_x)
            convolved feature maps
        """
        N = feature_maps.shape[0]
        I = feature_maps.shape[1]
        Y = feature_maps.shape[2]
        X = feature_maps.shape[3]
        J = kernels.shape[0]
        assert I == kernels.shape[1]
        Y_k = kernels.shape[2]
        X_k = kernels.shape[3]
        Y_o = (Y - 2 * (Y_k / 2)) / stride_y
        X_o = (X - 2 * (X_k / 2)) / stride_x
        A = np.zeros((N, J, Y_o, X_o))

        arguments = {"compiler": "gcc"}
        arguments.update(kwargs)
        compiler = arguments["compiler"]

        raise NotImplementedError("TODO implement fprop for conv. layer")

        code = """
        /*
        Available variabes
        ==================

        Input
        -----
        N                          - number of samples
        I                          - number of input feature maps
        J                          - number of output feature maps
        X                          - number of input feature map columns
        Y                          - number of input feature map rows
        X_o                        - number of output feature map columns
        Y_o                        - number of output feature map rows
        X_k                        - number of filter kernel columns
        Y_k                        - number of filter kernel rows
        bias(j, i)                 - the bias between the j-th output feature
                                     map and the i-th input feature map
        feature_maps(n, i, yi, xi) - the output of the i-th input feature map
                                     at position (xi, yi) for the n-th sample
        kernels(j, i, yk, xk)      - convolution kernel between i-th input feature
                                     map and j-th output feature map at position
                                     (xk, yk)

        Output
        ------
        A(n, j, yo, xo)            - the activation of the j-th output feature
                                     map at pixel (xo, yo) for the n-th sample
         */
        ////////////////////////////////////////////////////////////////////////////
        // TODO implement, see back_convolve() for an example
        ////////////////////////////////////////////////////////////////////////////
        """
        variables = ["N", "J", "I", "Y_o", "X_o", "stride_x", "stride_y", "Y_k",
                     "X_k", "feature_maps", "kernels", "bias", "A"]

        scipy.weave.inline(code, variables,
                           type_converters=scipy.weave.converters.blitz,
                           compiler=compiler)
        return A


    def back_convolve(feature_maps, kernels, bias, Deltas, stride_y=1, stride_x=1,
                      **kwargs):
        """Convolve I 2-dimensional feature_maps to generate J output feature_maps.

        Parameters
        ----------
        feature_maps : array-like, shape = (N, I, Y, X)
            input feature maps

        kernels : array-like, shape = (J, I, Y_k, X_k)
            filter kernels

        bias : array-like, shape = (J, I)
            bias matrix

        Deltas : array-like, shape = (N, J, Y-2*floor(Y_k/2)/stride_y,
                                      X-2*floor(X_k/2)/stride_x)
            deltas

        stride_y : int, optional
            column step size

        stride_x : int, optional
            row step size

        kwargs : dictionary, optional
            Optional arguments

        Returns
        -------
        der : array-like, shape = [J, I, Y_k, X_k]
            filter kernel derivatives

        derb : array-like, shape = [J, I]
            bias derivatives

        dEdX : array-like, shape = [N, I, Y, X]
            errors
        """
        N = feature_maps.shape[0]
        I = feature_maps.shape[1]
        Y = feature_maps.shape[2]
        X = feature_maps.shape[3]
        J = kernels.shape[0]
        assert I == kernels.shape[1]
        Y_k = kernels.shape[2]
        X_k = kernels.shape[3]
        Y_o = (Y - 2 * (Y_k / 2)) / stride_y
        X_o = (X - 2 * (X_k / 2)) / stride_x

        der = np.zeros_like(kernels)
        derb = np.zeros_like(bias)
        dEdX = np.zeros_like(feature_maps)

        arguments = {"compiler": "gcc"}
        arguments.update(kwargs)
        compiler = arguments["compiler"]

        code = """
        /*
        Available variabes
        ==================

        Input
        -----
        N                          - number of samples
        I                          - number of input feature maps
        J                          - number of output feature maps
        X                          - number of input feature map columns
        Y                          - number of input feature map rows
        X_o                        - number of output feature map columns
        Y_o                        - number of output feature map rows
        X_k                        - number of filter kernel columns
        Y_k                        - number of filter kernel rows
        feature_maps(n, i, yi, xi) - the output of the i-th input feature map
                                     at position (xi, yi) for the n-th sample
        kernels(j, i, yk, xk)      - convolution kernel between i-th input feature
                                     map and j-th output feature map at position
                                     (xk, yk)
        Deltas(n, j, yo, xo)       - derivative of the error function for the
                                     n-the sample with respect to the activation
                                     (xo, yo) from the j-th output feature map

        Output
        ------
        dEdX(n, i, yi, xi)         - derivative of the error function for the
                                     n-th sample with respect to the pixel
                                     (xi, yi) from the i-th input feature map
        der(j, i, yk, xk)          - derivative of the error function with
                                     respect to the value of the filter kernel
                                     between the j-th output feature map and the
                                     i-th input feature map at position (xk, yk),
                                     this is a sum over all samples
        derb(j, i)                 - derivative of the error function with
                                     respect to the bias between the j-th output
                                     feature map and the i-th input feature map,
                                     this is a sum over all samples
         */
        for(int n = 0; n < N; n++)
        {
            for(int j = 0; j < J; j++)
            {
                for(int i = 0; i < I; i++)
                {
                    for(int yo = 0, yi = 0; yo < Y_o; yo++, yi+=stride_y)
                    {
                        for(int xo = 0, xi = 0; xo < X_o; xo++, xi+=stride_x)
                        {
                            for(int yk = 0, yik = yi; yk < Y_k; yk++, yik++)
                            {
                                for(int xk = 0, xik = xi; xk < X_k; xk++, xik++)
                                {
                                    dEdX(n, i, yik, xik) += kernels(j, i, yk, xk) *
                                            Deltas(n, j, yo, xo);
                                    der(j, i, yk, xk) += Deltas(n, j, yo, xo) *
                                            feature_maps(n, i, yik, xik);
                                }
                            }
                            derb(j, i) += Deltas(n, j, yo, xo);
                        }
                    }
                }
            }
        }
        """
        variables = ["N", "J", "I", "Y_o", "X_o", "stride_x", "stride_y", "Y_k",
                     "X_k", "kernels", "feature_maps", "Deltas", "der", "derb",
                     "dEdX"]

        scipy.weave.inline(code, variables,
                           type_converters=scipy.weave.converters.blitz,
                           compiler=compiler)
        return der, derb, dEdX
