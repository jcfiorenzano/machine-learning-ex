# Description: Here we are going to use a very simple example of the use
# of a decision tree with small data set of features and label
# to predict if a determinate fruit it is a orange or an apple

# author: Juan Carlos Fiorenzano
# based on the Google Developers video tutorial :
# https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=4

from sklearn import tree

'''
    Due to scikit-learn only allow real values we have to make some
    transformations:

    The texture codification is:
        1: Smooth texture
        0: Bumpy texture

    The labels codification is:
        1: apple
        0: orange
'''
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [1, 1, 0, 0]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)
print(classifier.predict([[160, 0]]))
