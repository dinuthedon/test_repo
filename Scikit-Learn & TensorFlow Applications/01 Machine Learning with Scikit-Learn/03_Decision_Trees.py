#==============================================================================
""" Decision Trees - Iris Dataset - Breast Cancer dataset

Instructions:
    1. Load the Iris Dataset from SciKit-Learn
    2. Fit a Decision Tree classifier model 
    3. Print out accuracies on training and test dataset
    4. Refit the model by limiting maximum depth of the tree to 3
    4. Visualize the decision tree using plot_decision_tree object
        under adspy_shared_utilities (get the .py file from TA)
    5. Display importance of various features as horizontal barchart
    6. Visualize the class regions of the classifier 
        (TA will help with sourcecode)
    7. Load the Breast Cancer Dataset from SciKit-Learn
    8. Fit a Decision Tree Classifier
    9. Print out accuracies on training and test dataset
    10.Display importance of various features as horizontal barchart
    
Scikit-Learn Usage Reference: 
    Scikit-Learn: Machine Learning in Python,
    Pedregosa et al., JMLR 12, pp. 2825-2830, 2011 
@author: Dhineshkumar"""
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_decision_tree
from adspy_shared_utilities import plot_feature_importances
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot


#==============================================================================
# Loading and Iris  Dataset
#==============================================================================

iris = load_iris()

# Creating test-train split of the data
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 3)

#==============================================================================
# Decision Tree Classifier Model fitting
#==============================================================================

clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

# Refitting the model by limiting maximum depth of the tree to 3

clf2 = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)

print("Decision Tree Classifier model with max depth = 3:")
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf2.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf2.score(X_test, y_test)))

# Visualizing Decision Tree

#plot_decision_tree(clf, iris.feature_names, iris.target_names)

# Plotting the feature importance

print("Importance of features of the Iris dataset:")
plt.figure(figsize=(10,4), dpi=80)
plot_feature_importances(clf, iris.feature_names)
plt.show()

print('Feature importances: {}'.format(clf.feature_importances_))


from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 0)
fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
tree_max_depth = 4

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train
    
    clf = DecisionTreeClassifier(max_depth=tree_max_depth).fit(X, y)
    title = 'Decision Tree, max_depth = {:d}'.format(tree_max_depth)
    plot_class_regions_for_classifier_subplot(clf, X, y, None,
                                             None, title, axis,
                                             iris.target_names)
    
    axis.set_xlabel(iris.feature_names[pair[0]])
    axis.set_ylabel(iris.feature_names[pair[1]])
    
plt.tight_layout()
plt.show()

#==============================================================================
# Decision Tree Classifier Model fitting for Breast Cancer dataset
#==============================================================================

# Loading Breast cancer dataset for classification
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 8,
                            random_state = 0).fit(X_train, y_train)

#plot_decision_tree(clf, cancer.feature_names, cancer.target_names)

print('Breast cancer dataset Decision Tree')
print('Accuracy of DT classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of DT classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

# Plotting the feature importance

print("Feature importance of Breast cancer dataset:")
plt.figure(figsize=(10,6),dpi=80)
plot_feature_importances(clf, cancer.feature_names)
plt.tight_layout()

plt.show()

print('Feature importances: {}'.format(clf.feature_importances_))

#==============================================================================
