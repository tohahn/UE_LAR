# Implementation of a Neural Net

The code in this directory has been tested on Ubuntu 14.04 with Python 2.7
and Windows 8.1 with Anaconda and Python 3.4. If you have any problems
with your machine you should contact me (Alexander.Fabisch@dfki.de).

## Dependencies

In this implementation we assume that you have several packages installed
on your machine:

* NumPy
* SciPy
* Scikit-Learn

You probably want to use one of the following packages to implement
the convolution in exercise 11, otherwise it will be too slow.

* numba (recommended especially for Windows and/or Python 3)
* scipy.weave
* cython
* parakeet

If you have Anaconda installed you will already have NumPy, SciPy,
Scikit-Learn, numba, and cython installed.

## Files

### Exercise 10

* minibatch_sgd.py - implements mini-batch stochastic gradient descent,
  provides the class `MiniBatchSGD` that is derived from sklearn's
  `BaseEstimator` so that it can be used with model selection tools from
  sklearn
* multilayer_neural_network.py - implements a generic multilayer neural
  network that you have to implement in the exercise
* sarcos.py - downloads and loads the Sarcos dataset
* tools.py - contains some functions that help us to implement the neural
  net (e.g. activation functions)
* test_gradient.py - test script to check the gradients of the neural net
* train_sine.py - a script that trains a neural net on a toy dataset
* train_sarcos.py - a script that trains a neural net on the Sarcos dataset

### Exercise 11

* t10k-images-idx3-ubyte - contains the test images
* t10k-labels-idx1-ubyte - contains labels for the test set
* train-images-idx3-ubyte - contains the training images
* train-labels-idx1-ubyte - contains labels for the training set
* mnist.py - loads the MNIST dataset of handwritten digits and provides
  some utility functions
* train_mnist.py - a script that trains a neural net on the MNIST
  training set
* analize.ipynb - a notebook that loads the neural net that has been trained
  on the MNIST data and plots several filters, activations, etc. of the
  neural net