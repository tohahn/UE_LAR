#!/usr/bin/env python3

# do not use any other imports!
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KNeighborsClassifier as BlackBoxClassifier
from sklearn.datasets import load_iris

class Evaluation:
    """This class provides functions for evaluating classifiers """
        
    def generate_cv_pairs(self, n_samples, n_folds=5, n_rep=1, rand=False, 
                          y=None):
        """ Train and test pairs according to k-fold cross validation 

        Parameters
        ----------

        n_samples : int
            The number of samples in the dataset
        
        n_folds : int, optional (default: 5)
            The number of folds for the cross validation
        
        n_rep : int, optional (default: 1)
            The number of repetitions for the cross validation

        rand : boolean, optional (default: False)
            If True the data is randomly assigned to the folds. The order of the
            data is maintained otherwise. Note, *n_rep* > 1 has no effect if
            *random* is False.

        y : array-like, shape (n_samples), optional (default: None)
            If not None, cross validation is performed with stratification and 
            y provides the labels of the data. 

        Returns
        -------

        cv_splits : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split. The list has the length of
            *n_folds* x *n_rep*.

        """
        ### YOUR IMPLEMENTATION GOES HERE ###


    def apply_cv(self, X, y, train_test_pairs, classifier):
        """ Use cross validation to evaluate predictions and return performance 

        Apply the metric calculation to all test pairs

        Parameters
        ----------

        X : array-like, shape (n_samples, feature_dim)
            All data used within the cross validation

        y : array-like, shape (n-samples)
            The actual labels for the samples

        train_test_pairs : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split

        classifier : function
            Function that trains and tests a classifier and returns a 
            performance measure. Arguments of the functions are the training
            data, the testing data, the correct labels for the training data,
            and the correct labels for the testing data.

        Returns
        -------

        performance : float
            The average metric value across train-test-pairs
        """
        ### YOUR IMPLEMENTATION GOES HERE ###


    def black_box_classifier(self, X_train, X_test, y_train, y_test):
        """ Learn a model on the training data and apply it on the testing data

        Parameters
        ----------

        X_train : array-like, shape (n_samples, feature_dim)
            The data used for training 

        X_test : array-like, shape (n_samples, feature_dim)
            The data used for testing

        y_train : array-like, shape (n-samples)
            The actual labels for the training data
        
        y_test : array-like, shape (n-samples)
            The actual labels for the testing data

        neighbors : int, optional (default: 1)
            The number of neighbors considered for the classification

        Returns
        -------

        accuracy : float
            Accuracy of the model on the testing data
        """
        bbc = BlackBoxClassifier(n_neighbors=10)
        bbc.fit(X_train, y_train)
        acc = bbc.score(X_test, y_test)
        return acc

    def generate_roc_points(self, predictions, y_actual):
        """ Efficiently generating ROC points

        ### THE SUMMARY OF YOUR ALGORITHM GOES HERE ###

        Parameters
        ----------

        predictions : array-like, shape (n-samples)
            The classifier's estimates that the samples are positive

        y_actual : array-like, boolean-like, shape (n-samples)
            The actual labels for the samples

        Returns
        -------

        roc_points : list of tuples (false-positive rate, true-positive rate)
            ROC points increasing by false-positive rate
        """
        ### YOUR IMPLEMENTATION GOES HERE ###


    def auc(self, roc_points):
        """ Calculate the area under curve (AUC)
        
        ### THE SUMMARY OF YOUR ALGORITHM GOES HERE ###

        Parameters
        ----------

        roc_points : list of tuples (false-positive rate, true-positive rate)
            ROC points increasing by false-positive rate

        Returns
        -------

        area : float
            The area under curve (AUC)
        """
        ### YOUR IMPLEMENTATION GOES HERE ###


    def load_score_file(self, filename):
        """ Read file with structure instance number, actual class, classifier score
        
        Parameters
        ----------

        filename: string
            absolute path to the file that should be read in 

        Returns
        -------

        predictions : array-like, shape (n-samples)
            The classifier's estimates that the samples are positive
        
        y_actual : array-like, shape (n-samples)
            The actual labels for the samples
        """
        ### YOUR IMPLEMENTATION GOES HERE ###


if __name__ == '__main__':
    # Instance of the Evaluation class
    c = Evaluation()
        
    ### YOUR IMPLEMENTATION FOR PROBLEM 3.1 b) GOES HERE ###
    
    ## EXERCISE 3.3 c) ##
    # loading a data set for evaluation
    iris = load_iris()
    X = iris["data"][40:,:]
    y = iris["target"][40:]
    
    ### YOUR IMPLEMENTATION FOR EXERCISE 3.3 c) GOES HERE ###
