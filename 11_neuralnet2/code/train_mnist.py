"""Train multilayer neural network with MBSGD on MNIST data set."""
import numpy as np
import pickle
from mnist import read, scale, generate_targets, model_accuracy
from multilayer_neural_network import MultilayerNeuralNetwork
from minibatch_sgd import MiniBatchSGD


if __name__ == "__main__":
    # Load training set and test set
    categories = 10
    train_images, train_labels = read(range(categories), "training")
    test_images, test_labels = read(range(categories), "testing")
    # Scaling the input to from [0, 255] to [-1, 1] is necessary for saturating
    # activation functions (e.g. tanh, logistic sigmoid)
    train_images = scale(train_images)[:, None, :, :]
    test_images = scale(test_images)[:, None, :, :]
    # We use 1-of-c category encoding (c is the number of categories/classes)
    train_targets = generate_targets(train_labels)
    test_targets = generate_targets(test_labels)

    # Train neural network
    D = train_images.shape[1:]
    F = categories
    ############################################################################
    # Here you should define and train 'mlnn' (type: MultilayerNeuralNetwork)
    layers = \
        [
            {
                "type": "convolutional",
                "num_feature_maps": 6,
                "kernel_shape": (5, 5),
                "strides": (2, 2)
            },
            {
                "type": "convolutional",
                "num_feature_maps": 50,
                "kernel_shape": (5, 5),
                "strides": (2, 2)
            },
            {
                "type": "fully_connected",
                "num_nodes": 100
            }
        ]
    
    mlnn = MultilayerNeuralNetwork(
        D=(1, 28,28), F=10, layers=layers, std_dev=0.01,
        verbose=1)
    mbsgd = MiniBatchSGD(net=mlnn, epochs=15, batch_size=32, alpha=0.01, alpha_decay=0.9999, min_alpha=0.00005, eta=0.5, eta_inc=0.00001, max_eta=0.9, random_state=0, verbose=1)
    mbsgd.fit(train_images, train_targets)

    ############################################################################

    # Print accuracy and cross entropy on test set
    accuracy = 100 * model_accuracy(mlnn, test_images, test_labels)
    error = mlnn.error(test_images, test_targets)
    print("Accuracy on test set: %.2f %%" % accuracy)
    print("Error = %.3f" % error)

    # Store learned model, you can restore it with
    # mlnn = pickle.load(open("mnist_model.pickle", "rb"))
    # and use it in your evaluation script
    pickle.dump(mlnn, open("mnist_model.pickle", "wb"))
