#==============================================================================
""" Ensemble Learning - Random Forests

Instructions:
    1. Load the fruits dataset
    2. Model an ensemble learner using random forest classification
    3. Print out random forest classification regimes for different
        pairs of features in the dataset
    4. Print out classification accuracies on training and test
        datasets with n_estimators = 10
    5. Load the Breast Cancer Dataset 
    6. Print out accuracy of random forest classifier accuracies
        on classifying breast cancer dataset using max_features = 8

Scikit-Learn Usage Reference: 
    Scikit-Learn: Machine Learning in Python,
    Pedregosa et al., JMLR 12, pp. 2825-2830, 2011 
@author: Dhineshkumar"""
#==============================================================================

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from adspy_shared_utilities import plot_class_regions_for_classifier
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

#==============================================================================
# Ensemble Random Forest classificaton to Fruits dataset
#==============================================================================

# Loading fruits dataset

fruits = pd.read_table('data/fruit_data_with_colors.txt')

feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']


X_train, X_test, y_train, y_test = train_test_split(X_fruits.as_matrix(),
                                                   y_fruits.as_matrix(),
                                                   random_state = 0)
fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

title = 'Random Forest, fruits dataset, default settings'

pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]

for pair, axis in zip(pair_list, subaxes):
    
    X = X_train[:, pair]
    y = y_train
    
    clf = RandomForestClassifier().fit(X, y)
    plot_class_regions_for_classifier_subplot(clf, X, y, None,
                                             None, title, axis,
                                             target_names_fruits)
    
    axis.set_xlabel(feature_names_fruits[pair[0]])
    axis.set_ylabel(feature_names_fruits[pair[1]])
    
plt.tight_layout()
plt.show()

clf = RandomForestClassifier(n_estimators = 10,
                            random_state=0).fit(X_train, y_train)

print('Random Forest, Fruit dataset, default settings')
print('Accuracy of RF classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

#==============================================================================
# Breast Cancer Dataset dataset
#==============================================================================

# Loading Breast cancer dataset 
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer,
                                                    random_state=0)
# Fitting Random Forest classifier with max_features = 8

clf = RandomForestClassifier(max_features = 8, random_state = 0)

clf.fit(X_train, y_train)

print('Breast cancer dataset')
print('Accuracy of RF classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
#==============================================================================
