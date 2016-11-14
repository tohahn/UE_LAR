import numpy as np
from itertools import combinations

def correlation(a,b):
	a_mean = np.mean(a)
	b_mean = np.mean(b)
	return np.sum((a-a_mean) * (b-b_mean))/(np.sqrt(np.sum(np.power(a-a_mean, 2))) * np.sqrt(np.sum(np.power(b-b_mean, 2))))

def average_correlation(data, features, class_name):
    if (len(features) > 1):
        feature_correlation = 0
        for a,b in combinations(features, 2):
            feature_correlation += abs(correlation(data[a],data[b]))
        feature_correlation /= len(list(combinations(features,2)))
    else:
        feature_correlation = 1

    class_correlation = 0
    for f in features:
        class_correlation += abs(correlation(data[f], data[class_name]))
    class_correlation /= len(features)
    print("\nEvaluationg featureset {2}\nThe average feature feature correlation is {0}.\nThe average feature class correlation is: {1}.".format(feature_correlation, class_correlation, features))
    return (feature_correlation, class_correlation)

def merit(data, features, class_name):
    count = len(features)
    res = average_correlation(data, features, class_name)
    merrit = (count * res[1]) / (np.sqrt(count + count * (count - 1) * res[0]))
    print("The merit is: {0}".format(merrit))
    return merrit

if __name__ == "__main__":
	data = np.zeros(4, dtype=[('a', 'f8'), ('b', 'f8'), ('c', 'f8'), ('class', 'f8')])
	data[0] = (-2, -1, 0, -1)
	data[1] = (-2, -1, 1.4, -1)
	data[2] = (2, 1, -1, 1)
	data[3] = (-.2, -.1, -1, 1)
        
        merit(data, ['a'], 'class')
        merit(data, ['b'], 'class')
        merit(data, ['c'], 'class')
        merit(data, ['a','b'], 'class')
        merit(data, ['a','c'], 'class')
        merit(data, ['b','c'], 'class')
        merit(data, ['a','b','c'], 'class')
