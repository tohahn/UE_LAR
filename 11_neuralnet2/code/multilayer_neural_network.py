"""Multilayer Neural Network."""
import numpy as np
from tools import (softmax, linear, linear_derivative, relu, relu_derivative,
                   convolve, back_convolve)


class FullyConnectedLayer(object):
    """Represents a trainable fully connected layer.

    Parameters
    ----------
    I : int or tuple
        input shape (without bias)

    J : int; outputs

    g : function: array-like -> array-like
        activation function y = g(a)

    gd : function: array-like -> array-like
        derivative g'(a) = gd(y)

    std_dev : float
        standard deviation of the normal distribution that we use to draw
        the initial weights

    verbose : int, optional
        verbosity level
    """
    def __init__(self, I, J, g, gd, std_dev, verbose=0):
        self.I = np.prod(I) + 1  # Add bias component
        self.J = J
        self.g = g
        self.gd = gd
        self.std_dev = std_dev

        self.W = np.empty((self.J, self.I))

        if verbose:
            print("Fully connected layer (%d nodes, %d x %d weights)"
                  % (self.J, self.J, self.I))

    def initialize_weights(self, random_state):
        """Initialize weights randomly.

        Parameters
        ----------
        random_state : RandomState or int
            random number generator or seed
        """
        ######################################################################
        # Initialize weight matrices randomly: draw weights from a Gaussian
        # distribution
        self.W = self.std_dev * random_state.randn(self.J, self.I)
        ######################################################################

    def get_output_shape(self):
        """Get shape of the output.

        Returns
        -------
        shape : tuple
            shape of the output
        """
        return (self.J,)

    def forward(self, X):
        """Forward propagate the output of the previous layer.

        Parameters
        ----------
        X : array-like, shape = [N, I or self.I-1]
            input

        Returns
        -------
        Y : array-like, shape = [N, J]
            output
        """
        N = X.shape[0]
        D = np.prod(X.shape[1:])
        if D != self.I - 1:
            raise ValueError("shape = " + str(X.shape))

        ######################################################################
        # Bias will be the first component
        self.X = np.hstack((np.ones((N, 1)), X.reshape(N, D)))
        # Compute activations
        A = self.X.dot(self.W.T)
        # Compute output
        self.Y = self.g(A)
        ######################################################################
        return self.Y

    def backpropagation(self, dEdY):
        """Backpropagate errors of the next layer.

        Parameters
        ----------
        dEdY : array-like, shape = [N, J]
            errors from the next layer

        Returns
        -------
        dEdX : array-like, shape = [N, I or self.I - 1]
            errors from this layer

        Wd : array-like, shape = [J, self.I]
            derivatives of the weights
        """
        if dEdY.shape[1] != self.J:
            raise ValueError("%r != %r" % (dEdY.shape[1], self.J))

        ######################################################################
        # Derivative g'(a) = gd(y)
        Yd = self.gd(self.Y)  # We use a trick here: Yd will be 1 for ReLU
        # Component-wise multiplication!
        Deltas = Yd * dEdY
        # Deltas.T * X
        Wd = Deltas.T.dot(self.X)
        # Errors for previous layer
        dEdX = Deltas.dot(self.W)
        # Get rid of bias
        dEdX = dEdX[:, 1:]
        ######################################################################
        return dEdX, Wd

    def get_weights(self):
        """Get current weights.

        Returns
        -------
        W : array-like, shape = [J * I + 1 or self.I]
            weight matrix
        """
        return self.W.flat

    def set_weights(self, W):
        """Set new weights.

        Parameters
        ----------
        W : array-like, shape = [J * I + 1 or self.I]
            weight matrix
        """
        self.W = W.reshape((self.J, self.I))

    def num_weights(self):
        """Get number of weights.

        Returns
        -------
        K : int
            number of weights
        """
        return self.W.size

    def __getstate__(self):
        # This will be called by pickle.dump, so we remove everything that
        # requires too much memory
        d = dict(self.__dict__)
        if "X" in d:
            del d["X"]
        if "Y" in d:
            del d["Y"]
        return d


class ConvolutionalLayer(object):
    """Represents a trainable convolutional layer.

    Parameters
    ----------
    I : int
        input feature maps

    J : int
        output feature maps

    Y : int
        input feature map rows

    X : int
        input feature map columns

    kernel_shape : pair of ints
        kernel rows and kernel columns

    strides : pair of ints
        row stride and column stride

    g : function: array-like -> array-like
        activation function y = g(a)

    gd : function: array-like -> array-like
        derivative g'(a) = gd(y)

    std_dev : float
        standard deviation of the normal distribution that we use to draw the
        initial weights

    verbose : int, optional
        verbosity level
    """
    def __init__(self, I, J, Y, X, kernel_shape, strides, g, gd, std_dev,
                 verbose=0):
        self.I = I
        self.J = J
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.Y_o = (Y - 2 * (kernel_shape[0] // 2)) // strides[0]
        self.X_o = (X - 2 * (kernel_shape[1] // 2)) // strides[1]
        self.g = g
        self.gd = gd
        self.std_dev = std_dev

        self.n_inputs = self.I * Y * X
        self.n_outputs = self.J * self.Y_o * self.X_o

        self.weight_shape = (self.J, self.I, self.kernel_shape[0],
                             self.kernel_shape[1])
        self.W = np.empty(self.weight_shape)
        self.B = np.empty((self.J, self.I))

        if verbose:
            print("Convolutional layer (%d x %d x %d nodes, %d x %d x %d x %d"
                  " weights)" % (self.J, self.Y_o, self.X_o, self.J, self.I,
                                 self.kernel_shape[0], self.kernel_shape[1]))

    def initialize_weights(self, random_state):
        """Initialize weights randomly.

        Parameters
        ----------
        random_state : RandomState or int
            random number generator or seed
        """
        ######################################################################
        # Initialize weight matrices randomly: draw weights from a Gaussian
        # distribution
        self.W = random_state.normal(0, self.std_dev, self.weight_shape)
        ######################################################################

    def get_output_shape(self):
        """Get shape of the output.

        Returns
        -------
        shape : tuple
            shape of the output
        """
        return (self.J, self.Y_o, self.X_o)

    def forward(self, X):
        """Forward propagate the output of the previous layer.

        Parameters
        ----------
        X : array-like, shape = [N, I, Y, X]
            input

        Returns
        -------
        y : array-like, shape = [N, J, Y_o, X_o]
            output
        """
        if np.prod(X.shape[1:]) != self.n_inputs:
            raise ValueError("shape = %s, expected %d inputs"
                             % (X.shape, self.n_inputs))

        self.X = X
        A = convolve(X, self.W, self.B, stride_y=self.strides[0],
                     stride_x=self.strides[1])
        # Compute output
        self.Y = self.g(A)
        return self.Y

    def backpropagation(self, dEdY):
        """Backpropagate errors of the next layer.

        Parameters
        ----------
        dEdY : array-like, shape = [N, J, Y_o, X_o]
            errors from the next layer

        Returns
        -------
        dEdX : array-like, shape = [N, I, Y, X]
            errors from this layer

        Wd : array-like, shape = [J*I*kernel_shape[0]*kernel_shape[1]+J*I,]
            derivatives of the weights
        """
        if np.prod(dEdY.shape[1:]) != self.n_outputs:
            raise ValueError("shape = %s" % dEdY.shape)

        dEdY = dEdY.reshape((dEdY.shape[0], self.J, self.Y_o, self.X_o))
        # Derivative g'(a) = gd(y)
        Yd = self.gd(self.Y)
        # Component-wise multiplication!
        Deltas = Yd * dEdY
        self.Wd, self.Bd, dEdX = back_convolve(
            self.X, self.W, self.B, Deltas, self.strides[0], self.strides[1])
        return dEdX, np.concatenate((self.Wd.flat, self.Bd.flat))

    def get_weights(self):
        """Get current weights.

        Returns
        -------
        W : array-like, shape = [J*I*kernel_shape[0]*kernel_shape[1]+J*I]
            weight (+ bias) vector
        """
        return np.concatenate((self.W.flat, self.B.flat))

    def set_weights(self, W):
        """Set new weights.

        Parameters
        ----------
        W : array-like, shape = [J*I*(Y_o*X_o+1)]
            weight (+ bias) vector
        """
        # Separate weights and bias
        num_weights = np.prod(self.weight_shape)
        self.W = W[:num_weights].reshape(self.weight_shape)
        self.B = W[num_weights:].reshape((self.J, self.I))

    def num_weights(self):
        """Get number of weights.

        Returns
        -------
        K : int
            number of weights
        """
        return self.W.size + self.B.size

    def __getstate__(self):
        # This will be called by pickle.dump, so we remove everything that
        # requires too much memory
        d = dict(self.__dict__)
        if "X" in d:
            del d["X"]
        if "Y" in d:
            del d["Y"]
        return d


class MultilayerNeuralNetwork(object):
    """Multilayer Neural Network (MLNN).

    Parameters
    ----------
    D : int or tuple
        input shape

    F : int
        number of outputs

    layers : list of dicts
        layer definitions

    training : string
        must be either classification or regression and defines the
        activation function of the last layer as well as the error function

    std_dev : float
        standard deviation of the normal distribution that we use to draw
        the initial weights

    verbose : int, optional
        verbosity level
    """

    def __init__(self, D, F, layers, training="classification", std_dev=0.05,
                 verbose=0):
        self.D = D
        self.F = F

        # Initialize layers
        self.layers = []
        I = self.D
        for layer in layers:
            l = None
            if layer["type"] == "fully_connected":
                l = FullyConnectedLayer(
                    I, layer["num_nodes"], relu, relu_derivative, std_dev,
                    verbose)
                I = l.get_output_shape()
            elif layer["type"] == "convolutional":
                assert len(I) == 3
                l = ConvolutionalLayer(
                    I[0], layer["num_feature_maps"], I[1], I[2],
                    layer["kernel_shape"], layer["strides"], relu,
                    relu_derivative, std_dev, verbose)
                I = l.get_output_shape()
            else:
                raise NotImplementedException("Layer type '%s' is not "
                                              "implemented." % layer["type"])
            self.layers.append(l)
        if training == "classification":
            self.layers.append(FullyConnectedLayer(
                I, self.F, softmax, linear_derivative, std_dev, verbose))
            self.error_function = "ce"
        elif training == "regression":
            self.layers.append(FullyConnectedLayer(
                I, self.F, linear, linear_derivative, std_dev, verbose))
            self.error_function = "sse"
        else:
            raise ValueError("Unknown 'training': %s" % training)

    def initialize_weights(self, random_state):
        """Initialize weights randomly.

        Parameters
        ----------
        random_state : RandomState or int
            random number generator or seed
        """
        for layer in self.layers:
            layer.initialize_weights(random_state)

    def error(self, X, T):
        """Calculate the Cross Entropy (CE).

        .. math::

            E = -\sum_n \sum_f ln(y^n_f) t^n_f,

        where n is the index of the instance, f is the index of the output
        component, y is the prediction and t is the target.

        Parameters
        ----------
        X : array-like, shape = [N, D]
            each row represents an instance

        T : array-like, shape = [N, F]
            each row represents a target

        Returns
        -------
        E : float
            error: SSE for regression, cross entropy for classification
        """
        if len(X) != len(T):
            raise ValueError("Number of samples and targets must match")

        # Compute error of the dataset
        if self.error_function == "ce":
        ########################################################################
            return -np.sum(np.log(self.predict(X)) * T)
        ########################################################################
        elif self.error_function == "sse":
        ######################################################################
            return 0.5 * np.linalg.norm(self.predict(X) - T) ** 2
        ######################################################################

    def numerical_gradient(self, X, T, eps=1e-5):
        """Compute the derivatives of the weights with finite differences.

        This function can be used to check the analytical gradient
        numerically. The partial derivative of E with respect to w is
        approximated through

        .. math::

            \partial E / \partial w = (E(w+\epsilon) - E(w-\epsilon)) /
                                      (2 \epsilon) + O(\epsilon^2),

        where :math:`\epsilon` is a small number.

        Parameters
        ----------
        X : array-like, shape = [N, D]
            input

        T : array-like, shape = [N, F]
            desired output (target)

        eps : float, optional
            small number, you can make eps smaller to increase the accuracy
            of the differentiation until roundoff errors occur

        Returns
        -------
        wd : array-like, shape = [K,]
            weight vector derivative
        """
        w = self.get_weights()
        w_original = w.copy()
        wd = np.empty_like(w)
        for k in range(len(w)):
            w[k] = w_original[k] + eps
            self.set_weights(w)
            Ep = self.error(X, T)
            w[k] = w_original[k] - eps
            self.set_weights(w)
            Em = self.error(X, T)
            w[k] = w_original[k]
            wd[k] = (Ep - Em) / (2.0 * eps)
        self.set_weights(w_original)
        return wd

    def gradient(self, X, T, get_error=False):
        """Calculate the derivatives of the weights.

        Parameters
        ----------
        X : array-like, shape = [N, D]
            input

        T : array-like, shape = [N, F]
            desired output (target)

        Returns
        -------
        g : array-like, shape = [K,]
            gradient of weight vector

        e : float, optional
            error
        """
        Wds = []
        # Forward propagation
        Y = self.predict(X)
        # Backpropagation
        dEdY = Y - T
        for l in reversed(range(len(self.layers))):
            dEdY, Wd = self.layers[l].backpropagation(dEdY)
            Wds.insert(0, Wd)
        g = np.concatenate([Wds[l].flat for l in range(len(self.layers))])
        if get_error:
            if self.error_function == "ce":
                ##############################################################
                raise NotImplementedError(
                    "TODO implement MultilayerNeuralNetwork.gradient()")
                ##############################################################
            elif self.error_function == "sse":
                ##############################################################
                e = 0.5 * np.linalg.norm(Y - T) ** 2
                ##############################################################
            return g, e
        else:
            return g

    def get_weights(self):
        """Get current weight vector.

        Returns
        -------
        w : array-like, shape (K,)
            weight vector
        """
        return np.concatenate([self.layers[l].get_weights()
                               for l in range(len(self.layers))])

    def set_weights(self, w):
        """Set new weight vector.

        Parameters
        ----------
        w : array-like, shape=[K,]
            weight vector
        """
        i = 0
        for l in range(len(self.layers)):
            k = self.layers[l].num_weights()
            self.layers[l].set_weights(w[i:i + k])
            i += k

    def predict(self, X):
        """Predict values.

        Parameters
        ----------
        X : array-like, shape = [N, D]
            each row represents an instance

        Returns
        -------
        Y: array-like, shape = [N, F]
            each row represents a prediction
        """
        # Forward propagation
        ######################################################################
        for l in range(len(self.layers)):
            X = self.layers[l].forward(X)
        ######################################################################
        return X
