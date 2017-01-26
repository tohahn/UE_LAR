"""Train multilayer neural network with MBSGD on Sarcos data set."""
import numpy as np
import pickle
from sarcos import download_sarcos
from sarcos import load_sarcos
from sarcos import nMSE
from sklearn.preprocessing import StandardScaler
from multilayer_neural_network import MultilayerNeuralNetwork
from minibatch_sgd import MiniBatchSGD
from sklearn.linear_model import LinearRegression

def printnMSE(model, X, Y, name): 
    print(name)
    # Print nMSE on test set
    Y_pred = model.predict(X)
    for f in range(Y_pred.shape[1]):
        print("Dimension %d: nMSE = %.2f %%"
              % (f + 1, 100 * nMSE(Y_pred[:, f], Y[:, f])))

def LinReg(X, Y, X_test, Y_test):
    model = LinearRegression()
    model.fit(X, Y)
    print("## Linear Regression ##")
    printnMSE(model, X, Y, "Train data")
    printnMSE(model, X_test, Y_test, "Test data")
    return model

def NeuralNet(X, Y, X_test, Y_test):
    layers = [{"type":"fully_connected", "num_nodes": 50}]
    mlnn = MultilayerNeuralNetwork(D=21, F=7, layers=layers, training="regression", std_dev=0.01)
    model = MiniBatchSGD(net=mlnn, epochs=100, batch_size=32, alpha=0.005, eta=0.5, random_state=0, verbose=0)
    model.fit(X,Y)
    print("## Neural Net ##")
    printnMSE(model, X, Y, "Train data")
    printnMSE(model, X_test, Y_test, "Test data")
    return model

if __name__ == "__main__":
    np.random.seed(0)

    # Download Sarcos dataset if this is required
    #download_sarcos()
    
    # Load training set and test set
    X, Y = load_sarcos("train")
    X_test, Y_test = load_sarcos("test")
    # Scale targets
    target_scaler = StandardScaler()
    Y = target_scaler.fit_transform(Y)
    Y_test = target_scaler.transform(Y_test)

    # Train model (code for exercise 10.2 1/2/3)
    model = LinReg(X, Y, X_test, Y_test)
    model = NeuralNet(X, Y, X_test, Y_test)
    # Store learned model, you can restore it with
    # model = pickle.load(open("sarcos_model.pickle", "rb"))
    # and use it in your evaluation script
    pickle.dump(model, open("sarcos_model.pickle", "wb"))
