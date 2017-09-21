#==============================================================================
""" Naive Bayes Classifier for Breast Cancer Dataset

Instructions:
    1. Generate two different synthetic datasets for Naive Bayes modeling
        (Sourcecode will be provided by TA)
    2. Use Scikit-Learn's functions to create a default train-test split
    3. Fit Gaussian Naive Bayes Classifier learner to the datasets
    4. Plot the classifier outpout
    5. Load the Breast Cancer Dataset 
    6. Print out accuracy of Naive Bayes classifier 
        on classifying breast cancer dataset

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
from sklearn.naive_bayes import GaussianNB
from adspy_shared_utilities import plot_class_regions_for_classifier

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

#==============================================================================
# Naive Bayes Classifier modeling to synthetic dataset1
#==============================================================================

# synthetic dataset for classification (binary)

plt.figure()
plt.title('Sample binary classification problem with two informative features')

X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,
                                n_redundant=0, n_informative=2,
                                n_clusters_per_class=1, flip_y = 0.1,
                                class_sep = 0.5, random_state=0)
plt.scatter(X_C2[:, 0], X_C2[:, 1], marker= 'o',
           c=y_C2, s=50, cmap=cmap_bold)

plt.show()

# Naive Bayes Model fitting to the synthetic dataset

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)

nbclf = GaussianNB().fit(X_train, y_train)

plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test,
                                 'Gaussian Naive Bayes classifier: Dataset 1')

plt.show()
#==============================================================================
# Naive Bayes Classifier modeling to synthetic dataset2 
# classes not linearly seperable
#==============================================================================


X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
y_D2 = y_D2 % 2

plt.figure()

plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2,
           marker= 'o', s=50, cmap=cmap_bold)

plt.show()

# Naive Bayes Model fitting to the synthetic dataset

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2,
                                                   random_state=0)

nbclf = GaussianNB().fit(X_train, y_train)
plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test,
                                 'Gaussian Naive Bayes classifier: Dataset 2')


#==============================================================================
# Breast Cancer Dataset dataset
#==============================================================================

# Loading Breast cancer dataset 
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer,
                                                    random_state=0)

nbclf = GaussianNB().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of GaussianNB classifier on training set: {:.2f} %'
     .format(nbclf.score(X_train, y_train)*100))
print('Accuracy of GaussianNB classifier on test set: {:.2f} %'
     .format(nbclf.score(X_test, y_test)*100))