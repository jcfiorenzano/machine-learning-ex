# description: in this example we are using a more populate data set and we extracted
# from them a little data set to make a test sample
#
# author: Juan Carlos Fiorenzano
# based on the Google Developers video tutorial :
# https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=4
# data set take it from https://en.wikipedia.org/wiki/Iris_flower_data_set

import numpy as np

from sklearn.datasets import load_iris
from sklearn import tree


iris = load_iris()
test_idx = [0, 50, 100]  # data index for testing the classifier, one example of each type of flower

# training data
# we select all the data from the data set except the testing data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)  # training the classifier

# testing the classifier, the two outputs has to be the same
print(test_target)
print(clf.predict(test_data))



