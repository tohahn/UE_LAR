import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from gradient_descent import gradient_descent


def load_dataset(filename="regression_dataset_1.txt"):
    """Load the given dataset.

    Parameters
    ----------
    filename : string, optional
        Name of the file that contains the data

    Returns
    -------
    X : array, shape (n_samples, n_features + 1)
        Inputs, extended by the "bias feature" which is always 1

    y : array, shape (n_samples,)
        Desired outputs
    """
    x, y = eval(open(filename, "r").read())
    n_samples = len(x)
    X = np.vstack((np.ones(n_samples), x)).T
    y = np.asarray(y)
    return X, y


def predict(w, X):
    """Predict outputs of a linear model.

    Parameters
    ----------
    w : array, shape (n_features + 1,)
        Weights and bias

    X : array, shape (n_samples, n_features + 1)
        Inputs, extended by the "bias feature" which is always 1

    Returns
    -------
    y : array, shape (n_samples,)
        Outputs
    """
    raise NotImplementedError("predict")


def sse(w, X, y):
    """Compute the sum of squared error of a linear model.

    Parameters
    ----------
    w : array, shape (n_features + 1,)
        Weights and bias

    X : array, shape (n_samples, n_features + 1)
        Inputs, extended by the "bias feature" which is always 1

    y : array, shape (n_samples,)
        Desired outputs

    Returns
    -------
    SSE : float
        Sum of squared errors
    """
    raise NotImplementedError("sse")


def dSSEdw(w, X, y):
    """Compute the gradient of the sum of squared error.

    Parameters
    ----------
    w : array, shape (n_features + 1,)
        Weights and bias

    X : array, shape (n_samples, n_features + 1)
        Inputs, extended by the "bias feature" which is always 1

    y : array, shape (n_samples,)
        Desired outputs

    Returns
    -------
    g : array, shape (n_features + 1,)
        Sum of squared errors
    """
    raise NotImplementedError("dSSEdw")


if __name__ == "__main__":
    X, y = load_dataset()

    # 'partial' will create new function-like objects that only have one
    # argument. The other arguments will contain our dataset.
    f = partial(sse, X=X, y=y)
    grad = partial(dSSEdw, X=X, y=y)

    plt.figure()
    ax = plt.subplot(111)

    for alpha in [0.0001, 0.001, 0.002, 0.0025]:
        w_star, path = gradient_descent(
            x0=[0.0, -0.5],
            alpha=alpha,
            grad=grad,
            n_iter=100,
            return_path=True
        )
        print("alpha = %g:\t w* = %r, SSE(w*) = %g"
              % (alpha, w_star, sse(w_star, X, y)))
        ax.semilogy(range(len(path)), [sse(w, X, y) for w in path],
                    label=r"$\alpha = %g$" % alpha)

    ax.legend(loc="upper right")
    plt.setp(ax, ylim=((0, 800)), xlabel="Iteration", ylabel="SSE")

    plt.show()
