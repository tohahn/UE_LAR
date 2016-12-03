import numpy as np
from sklearn.model_selection import KFold


class Classification:
    """This class holds different classification algorithms and the cv function
    """

    def apply_k_fold_cv(self, X, y, classifier=None, n_folds=5, **kwargs):
        """K fold cross validation.

        Parameters
        ----------
        X : array-like, shape (n_samples, feature_dim)
            The data for the cross validation

        y : array-like, shape (n-samples, label_dim)
            The labels of the data used in the cross validation

        classifier : function
            The function that is used for classification of the training data

        n_splits : int, optional (default: 5)
            The number of folds for the cross validation

        kwargs :
            Further parameters that get used e.g. by the classifier

        Returns
        -------
        accuracies : array, shape (n_splits,)
            Vector of classification accuracies for the n_splits folds.
        """
        assert X.shape[0] == y.shape[0]

        if len(X.shape) < 2:
            X = np.atleast_2d(X).T
        if len(y.shape) < 2:
            y = np.atleast_2d(y).T

        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []

        for train_index, test_index in cv.split(X):
            train_data = X[train_index, :]
            train_label = y[train_index, :]
            test_data = X[test_index, :]
            test_label = y[test_index, :]

            score = classifier(train_data, test_data,
                               train_label, test_label, **kwargs)

            scores.append(score)

        return np.array(scores)

    def kNN_classifier(self, X_train, X_test, y_train, y_test,
                       neighbors=1, metric=None, **kwargs):
        """K nearest neighbor classifier.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, feature_dim)
            The data for the training of the classifier

        X_test : array-like, shape (n_samples, feature_dim)
            The data for the test of the classifier

        y_train : array-like, shape (n-samples, label_dim)
            The labels for the training of the classifier

        y_test : array-like, shape (n-samples, label_dim)
            The labels for the test of the classifier

        neighbors : int, optional (default: 1)
            The number of neighbors considered for the classification

        metric : function
            The function that is used as a metric for the kNN classifier

        Returns
        -------
        accuracy : double
            Accuracy of the correct classified test data
        """

        ### YOUR IMPLEMENTATION GOES HERE ###

    def normalized_euclidean_distance(self, data_a, data_b, **kwargs):
        """Normalized euclidean distance metric"""

        ### YOUR IMPLEMENTATION GOES HERE ###

    def distance_measure_of_your_choice_1(self, data_a, data_b, **kwargs):
        """Chebyshev distance metric"""

        ### YOUR IMPLEMENTATION GOES HERE ###

    def distance_measure_of_your_choice_2(self, data_a, data_b, **kwargs):
        """Distance metric of your choice"""

        ### YOUR IMPLEMENTATION GOES HERE ###

    def bayesian_classifier(self, X_train, X_test, y_train, y_test, **kwargs):
        """Naive Bayes classifier.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, feature_dim)
            The data for the training of the classifier

        X_test : array-like, shape (n_samples, feature_dim)
            The data for the test of the classifier

        y_train : array-like, shape (n-samples, label_dim)
            The labels for the training of the classifier

        y_test : array-like, shape (n-samples, label_dim)
            The labels for the test of the classifier

        Returns
        -------
        accuracy : double
            Accuracy of the correct classified test data
        """

        ### YOUR IMPLEMENTATION GOES HERE ###

    def calc_prior_mean_variance(self, X, y):
        """Calculates the prior per class, the mean and the variance
           of the features per class.

        Parameters
        ----------
        X : array-like, shape (n_samples, feature_dim)
            The data used for training

        y : array-like, shape (n-samples, label_dim)
            The labels for the training data

        Returns
        -------
        accuracy : tuple of arrays
            Tuple containing the priors, means and variances
        """

        ### YOUR IMPLEMENTATION GOES HERE ###

if __name__ == '__main__':

    # Instance of the Classification class holding the distance metrics and
    # classification algorithm
    c = Classification()

    ### YOUR IMPLEMENTATION FOR EXERCISE 1 (b) GOES HERE ###

    ### YOUR IMPLEMENTATION FOR EXERCISE 1 (c) GOES HERE ###

    ### USE YOUR IMPLEMENTATION FOR EXERCISE 3 (a) HERE ###

    ### YOUR IMPLEMENTATION FOR EXERCISE 3 (b) GOES HERE ###
