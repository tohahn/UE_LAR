import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from collections import Counter
import matplotlib.pyplot as plt
import math


class DecisionTree():
    """ Representing a decision tree class.
    """
    def __init__(self):
        self.tree = None

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

    def dt_classifier(self, X_train, X_test, y_train, y_test,
                      attr_names=None, max_depth=-1, **kwargs):
        """ Decision tree implementation.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, feature_dim)
            The data to train the classifier

        X_test : array-like, shape (n-samples, label_dim)
            The data to test the learned classifier

        y_train : array-like, shape (n_samples, 1)
            The labels for the training data

        y_test : array-like, shape (n_samples, 1)
            The labels for the test data

        attr_names : list of tuple
            A list of tuples, where the first element is the index and the
            second the name of the attribute corresponding to the data

        max_depth : int, optional (default: -1)
            The maximal depth of the decision tree

        kwargs :
            Further parameters that get used e.g. by the classifier

        Returns
        -------
        accuracy : double
            Accuracy of the correct classified test data
        """
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        assert len(attr_names) > 0

        if max_depth == -1 or max_depth > len(attr_names):
            max_depth = len(attr_names)

        # Additionally work with a list of indices.
        idx = np.arange(X_train.shape[0])

        # First build the classifier
        dtn = DecisionTreeNode()
        self.tree = dtn.make_tree(idx, X_train, y_train, attr_names, max_depth)

        ### YOUR IMPLEMENTATION GOES HERE ###
        ### Use the learned classifier to evaluate the performance
        ### on the test set. Return as written in the specification
        true = 0
        false = 0
        
        for x,y in zip(X_test, y_test):
            pred = dtn.classify(x)
            if (pred == y):
                true += 1
            else:
                false += 1

        return true / (true+false)


class DecisionTreeNode():
    """ Class to represent a tree node.
        Leaves are empty trees that only have a label and no children
    """
    def __init__(self):
        self.label = None
        self.attribute = None
        self.attribute_value = None
        self.children = []  # List of DecisionTreeNodes

    def calc_impurity(self, class_labels):
        """ Calculate the impurity for a subset of the data

        Parameters
        ----------
        class_labels : array-like, shape (n_labels,)
            A list of class labels to calculate the impurity for.

        Returns
        -------
        impurity : double
            Impurity for the set of given class labels
        """

        ### YOUR IMPLEMENTATION GOES HERE ###
        ### Decide for an impurity measure of your choice.

    def classify(self, data_vector):
        """ Recursive method to assign a label to a given data point.

        Parameters
        ----------
        data_vector : array-like, shape (n_features,)
            One data point to be classified

        Returns
        -------
        label :
            The label for the classified data point
        """

        ### YOUR IMPLEMENTATION GOES HERE ###
        ### Use the tree structure to recursively go through
        ### all decisions for the tree
        ### If the attribute value was not in the training set
        ### weight each possible branch uniform and return the
        ### most common label.
        if not self.children:
            return self.label
        exp = data_vector[self.attribute]

        if exp not in [c.attribute_value for c in self.children]:
            return Counter([c.classify(data_vector) for c in self.children]).most_common(1)[0][0]
        
        for c in self.children:
            if (c.attribute_value == exp):
                return c.classify(data_vector)



    def make_tree(self, idx_lst, data, label, attributes, max_depth=0,
                  default=None):
        """ Create the decision tree recursively for the given data.

        Parameters
        ----------
        idx_lst : array-like, shape (n_samples,)
            A list of indices to represent a certain subset of the data

        data : array-like, shape (n_samples, n_features)
            The data that is to used to train the decision tree

        label : array-like, shape (n_samples,)
            The labels for all the training data.

        attributes : list of tuple
            A list of tuples, where the first element is the index and the
            second the name of the attribute corresponding to the data

        max_depth : int, optional (default: -1)
            The maximal depth of the decision tree to be formed

        default : label
            The default label for the case no decision can be made.

        Returns
        -------
        self : object
            Reference to a tree node structure representing a (sub)tree.
        """

        ### YOUR IMPLEMENTATION GOES HERE ###
        ### Decide on which attribute to split the data
        ### referenced by the idx_list and create corresponding
        ### subtrees. Build recursivly the whole decision tree
        ### up to the desired maximum depth
        if (len(attributes) == 0):
            self.label = Counter([label[i] for i in idx_lst]).most_common(1)[0][0]
            return self
        if (max_depth == 0):
            self.label = Counter([label[i] for i in idx_lst]).most_common(1)[0][0]
            return self
        if (max_depth > -1):
            max_depth -= 1
        
        info = self.info(*self.count_whole(idx_lst, label))
        infos = [self.comp_gain(info, self.partition(idx_lst, data, label, a)) for a in attributes]
        a_index = np.argmax(infos)

        a = attributes[a_index]
        new_attribs = attributes[:]
        new_attribs.remove(a)
        self.attribute = a[0]
        exps = list(set([data[i][a[0]] for i in idx_lst]))
        idx_lsts = {}
        for exp in exps:
            lst = []
            for i in idx_lst:
                if (data[i][a[0]] == exp):
                    lst.append(i)
            idx_lsts[exp] = lst
        
        self.children = [DecisionTreeNode() for exp in exps]
        for i,x in enumerate(exps):
            self.children[i].make_tree(idx_lsts[exp], data, label, new_attribs, max_depth, default)
        for i,x in enumerate(exps):
            self.children[i].attribute_value = x
        return self
        
    def info(self, v, g, a, n):
        w = v + g + a + n
        v_v = 0
        g_v = 0
        a_v = 0
        n_v = 0
        if (v > 0):
            v_v = v/w * math.log(v/w, 2)
        if (g > 0):
            g_v = g/w * math.log(g/w, 2)
        if (a > 0):
            a_v = a/w * math.log(a/w, 2)
        if (n > 0):
            n_v = n/w * math.log(n/w, 2)
        return -(v_v + g_v + a_v + n_v)
    
    def comp_gain(self, info, attrib):
        whole = sum([sum(d.values()) for d in attrib[1:]])
        infos = []
        for exp in attrib[0]:
            count = sum([d[exp] for d in attrib[1:]])
            infos.append(count / whole * self.info(attrib[1][exp], attrib[2][exp], attrib[3][exp], attrib[4][exp]))
        return info - sum(infos)

    def count_whole(self, idx_list, label):
        v = 0
        g = 0
        a = 0
        n = 0
        
        for i in idx_list:
            lbl = label[i]
                        
            if (lbl == "vgood"):
                v += 1
            elif (lbl == "good"):
                g += 1
            elif (lbl == "acc"):
                a += 1
            else:
                n += 1
        
        return (v, g, a, n)


    def partition(self, idx_list, data, label, attribute):
        exps = set([])
        vgood = {}
        good = {}
        acc = {}
        nacc = {}
        
        for i in idx_list:
            exp = data[i][attribute[0]] 
            if exp not in exps:
                exps.add(exp)
                vgood[exp] = 0
                good[exp] = 0
                acc[exp] = 0
                nacc[exp] = 0
            lbl = label[i]
            if (lbl == "vgood"):
                vgood[exp] += 1
            elif (lbl == "good"):
                good[exp] += 1
            elif (lbl == "acc"):
                acc[exp] += 1
            else:
                nacc[exp] += 1

        return (exps, vgood, good, acc, nacc)



def read_in_data(filename):

    names = ["buying", "maint", "doors", "persons", "lug_boot",
             "safety", "Class"]
    data = pd.read_csv(filename, names=names)
    data = data.dropna()

    attribute_names = list(data.dtypes.index[:-1])

    data = data.as_matrix()
    labels = data[:, -1]
    data = data[:, :-1]

    return (data, labels, list(enumerate(attribute_names)))


if __name__ == '__main__':
    (data, labels, attr_names) = read_in_data("../data/car.data")
    dt = DecisionTree()

    ### YOUR IMPLEMENTATION GOES HERE ###
    av_accuracies = [np.mean(dt.apply_k_fold_cv(data, labels, dt.dt_classifier, 10, attr_names=attr_names, max_depth=d)) for d in range(1,8)]
    print(av_accuracies)
    plt.plot(av_accuracies)
    plt.ylabel("Average Accuracy")
    plt.xlabel("Height of tree")
    plt.show()

    tree = DecisionTreeNode()
    tree.make_tree(np.arange(data.shape[0]), data, labels, attr_names, 2)
    print("[{0}](None)".format(attr_names[tree.attribute][1]))
    new_childs = []
    for c in tree.children:
        print("|[{0}]({1})".format(attr_names[c.attribute][1], c.attribute_value))
        new_childs.extend(c.children)
    for c in new_childs:
        print("||={0}".format(c.label))
