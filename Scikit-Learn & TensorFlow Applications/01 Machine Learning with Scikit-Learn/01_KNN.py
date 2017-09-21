#==============================================================================
""" Classification of Fruits Dataset based on K-Nearest-Neighbors

Instructions:
    1. Load the fruit_data_with_colors.txt dataset
    2. Use Scikit-Learn's functions to create a default train-test split
    3. Fit a 5-Nearest-Neighbors model for classification
    4. Predict the model classification for some random fruit data input
    5. Plot the variation of accuracy with respect to 'K'
    6. Plot the variation of accuracy with respect ot test-train split

Scikit-Learn Usage Reference: 
    Scikit-Learn: Machine Learning in Python,
    Pedregosa et al., JMLR 12, pp. 2825-2830, 2011 
@author: Dhinesh"""
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#==============================================================================
""" Loading and splitting  Dataset """

# Load the dataset
fruits = pd.read_table('data/fruit_data_with_colors.txt')

# Creating test-train split of the data
X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Default test-train split is 75-25

print("Dimensions of Train dataset: {} \n ".format(X_train.shape))

#==============================================================================
""" KNN model fitting""" 

#5NN Classifier model fitting
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)

print("The accuracy of the 5-NN classifier on test data: {0:.2f} %\n"
      .format(knn.score(X_test, y_test)*100))

#==============================================================================
""" KNN model predictions""" 

# create a dictionary mapping from fruit label value to fruit name 
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), 
                             fruits.fruit_name.unique()))   
print("Dictionary mapping different fruits in the dataset: \n {} \n"
      .format(lookup_fruit_name))

# Predicting new fruit classifications

fruit_prediction = knn.predict([[20, 4.3, 5.5]])
print("A fruit with a mass of 20g, width 4.3 cm, height 5.5 cm is classified as: {}"
      .format(lookup_fruit_name[fruit_prediction[0]]))

fruit_prediction = knn.predict([[100, 6.3, 8.5]])
print("A fruit with a mass of 100g, width 6.3 cm, height 8.5 cm is classified as: {}"
      .format(lookup_fruit_name[fruit_prediction[0]]))

#==============================================================================
""" Variation in Accuracy with respect to 'K' """

k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k', fontsize = 14)
plt.ylabel('Accuracy', fontsize = 14)
plt.scatter(k_range, scores)
plt.title('Accuracy vs K', fontsize = 14)
plt.xticks([0,5,10,15,20]);
plt.show()

#==============================================================================
""" Variation in Accuracy with respect to test-train split """

split = [0.9, 0.75, 0.6, 0.5,0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in split:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)', fontsize = 14)
plt.ylabel('Accuracy', fontsize = 14)
plt.title('Accuracy vs Test-Train split', fontsize = 14)
plt.show()

#==============================================================================
