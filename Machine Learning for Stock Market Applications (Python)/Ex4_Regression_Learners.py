#==============================================================================
"""Regression Learner Implementation:
    1. Random Tree Learner
    2. Linear Regression Learner
    3. Polynomial Regression Learner
    4. Ensemble Learner - Bootstrap Aggregatiion using Random Tree Learner 
 
@author: Dhinesh"""
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

#==============================================================================
# FUNCTION DEFINITIONS: 
#==============================================================================
"""Function that returns test - train split"""
def test_train(data):
    y = data['EM']
    X = data[['ISE','SP','DAX','FTSE','NIKKEI','BOVESPA','EU']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def test_train_poly(data):
    y = data['EM']
    X = data[['ISE','SP','DAX','FTSE','NIKKEI','BOVESPA','EU']]
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=0)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def RTLearner(X_train, X_test, y_train, y_test):
    reg = RandomForestRegressor(max_features = 7, random_state = 0)
    reg.fit(X_train,y_train)
    
    print('Accuracy of RF Regressor on training set: {:.2f}'
          .format(reg.score(X_train, y_train)))
    print('Accuracy of RF Regressor on test set: {:.2f}'
          .format(reg.score(X_test, y_test)))
    

def LinRegLearner(X_train, X_test, y_train, y_test):
    linreg = Ridge().fit(X_train, y_train)

    print('linear model coeff (w): {}'
          .format(linreg.coef_))
    print('linear model intercept (b): {:.3f}'
          .format(linreg.intercept_))
    print('R-squared score (training): {:.3f}'
          .format(linreg.score(X_train, y_train)))
    print('R-squared score (test): {:.3f}'
          .format(linreg.score(X_test, y_test)))
    
def PolyRegLearner(X_train, X_test, y_train, y_test):
    linreg = LinearRegression().fit(X_train, y_train)

    print('(poly deg 2) linear model coeff (w):\n{}'
          .format(linreg.coef_))
    print('(poly deg 2) linear model intercept (b): {:.3f}'
          .format(linreg.intercept_))
    print('(poly deg 2) R-squared score (training): {:.3f}'
          .format(linreg.score(X_train, y_train)))
    print('(poly deg 2) R-squared score (test): {:.3f}\n'
          .format(linreg.score(X_test, y_test)))

def BagLearner(data, num_bags):
    y = data['EM']
    X = data[['ISE','SP','DAX','FTSE','NIKKEI','BOVESPA','EU']]
    size = len(y)
    train_size = int(0.8 * size)
    validation_size = size - train_size
   
    for i in range(num_bags):
        # Create a randomized index into the full / combined training-set.
        idx = np.random.permutation(size)

        idx_train = idx[0:train_size]
        idx_validation = idx[train_size:]

        x_train_new = X[idx_train]
        y_train_new = y[idx_train]

        x_validation = X[idx_validation]
        y_validation = y[idx_validation] 
        
        print("Bag {} execution: \n".format(i))
        
        print(x_train_new)
        print(y_train_new)
        print(x_validation)
        print(x_validation)
        
        RTLearner(x_train_new, X_test, y_train_new, y_test)
    

#==============================================================================
# REGRESSION LEARNER:    
#==============================================================================

def Regression():
    #Read in Data
    data = pd.read_csv("data/ISTANBUL.csv")
    
    X_train, X_test, y_train, y_test = test_train(data)

    RTLearner(X_train, X_test, y_train, y_test)
    
    LinRegLearner(X_train, X_test, y_train, y_test)
    
    X_train_poly, X_test_poly, y_train_poly, y_test_poly = test_train_poly(data)
    
    PolyRegLearner(X_train_poly, X_test_poly, y_train_poly, y_test_poly)
    
    num_bags = 2
    
    BagLearner(data, num_bags)

    
x = Regression()