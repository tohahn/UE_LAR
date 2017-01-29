"""MNIST data set."""
import os
import struct
import array
import numpy as np


def read(digits, dataset="training", path="."):
    """Loads MNIST files into 3D numpy arrays.

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py

    MNIST: http://yann.lecun.com/exdb/mnist/

    Parameters
    ----------
    digits : list
        digits we want to load

    dataset : string
        'training' or 'testing'

    path : string
        path to the data set files
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    struct.unpack(">II", flbl.read(8))
    lbl = array.array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array.array("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ind[i]*rows*cols:(ind[i]+1)*rows*cols]
                             ).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def scale(x):
    """Scales values to [-1, 1].

    Parameters
    ----------
    x : array-like, shape = arbitrary
        unscaled data

    Returns
    -------
    x_scaled : array-like, shape = x.shape
        scaled data
    """
    minimum = x.min()
    return 2.0 * (x - minimum) / (x.max() - minimum) - 1.0


def generate_targets(labels):
    """1-of-c category encoding (c is the number of categories/classes).

    Parameters
    ----------
    labels : array-like, shape = [N, 1]
        class labels (0 to c)

    Returns
    -------
    targets : array-like, shape = [N, c]
        target matrix with 1-of-c encoding
    """
    N = len(labels)
    classes = labels.max() + 1
    targets = np.zeros((N, classes))
    for n in range(N):
        targets[n, labels[n]] = 1.0
    return targets


def model_accuracy(model, X, labels):
    """Compute accuracy of the model on data set.

    Parameters
    ----------
    model : Model
        learned model (has to support the predict function)

    X : array-like, shape = [N, D]
        inputs (scaled to [-1, 1])

    labels : array-like, shape = [N, 1]
        class labels (lie within [0, c-1])

    Returns
    -------
    accuracy : float
        fraction of correct predicted labels, range [0, 1]
    """
    predicted = np.argmax(model.predict(X), axis=1)[:, None]
    return float((predicted == labels).sum()) / len(labels)
