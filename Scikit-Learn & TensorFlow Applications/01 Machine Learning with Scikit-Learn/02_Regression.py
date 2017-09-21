#==============================================================================
""" Regression Models for a synthetic dataset

Instructions:
    1. Create a synthetic dataset with the following commands:
        >from sklearn.datasets import make_regression
        >X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
    2. Plot the random dataset
    3. Use Scikit-Learn's functions to create a default train-test split

Part-1: KNN Regression
    1. Fit a Linear Regression model for the synthetic dataset
    2. Compute R^2 score of the test data based on the trained KNN model
    3. Plot the target values for K=1 and K=3
    
Part-2: Linear Regression
    1. Fit a Linear Model for the synthetic dataset
    2. Print out the model parameters and plot the fitted line
    
Part-3: Ridge Regression
    1. Fit a Ridge Regression model for the synthetic dataset
    2. Print out the model parameters
    3. Apply feature scaling (MinMaxScaler) and compute model parameters
    4. Print out the effect of varying the learning rate alpha

Part-4: Lasso Regression
    1. Fit a Lasso Regression model for the scaled synthetic dataset
    2. Print out the model parameters
    3. Print out the effect of varying the learning rate alpha



Scikit-Learn Usage Reference: 
    Scikit-Learn: Machine Learning in Python,
    Pedregosa et al., JMLR 12, pp. 2825-2830, 2011 
@author: Dhineshkumar"""
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso

#==============================================================================
# Creating synthetic dataset
#==============================================================================

#Dataset creaation
X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)

plt.figure()
plt.title('Synthetic dataset for Regression', fontsize = 14)
plt.scatter(X_R1, y_R1, marker= 'o', s=50)
plt.show()

#==============================================================================
# KNN Regression Learner
#==============================================================================

# Test-Train splitting

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, 
                                                    random_state = 0)
# KNN Regressor Model Fitting

knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)

# Prediction on test set
print("The predicted values on the test set: \n {}"
      .format(knnreg.predict(X_test)))

print("R-squared value of test data: {:.3f}"
     .format(knnreg.score(X_test, y_test)))

# Target values for K=1 and K=3

fig, subaxes = plt.subplots(1, 2, figsize=(8,4))

X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], 
                                                    random_state = 0)

for thisaxis, K in zip(subaxes, [1, 3]):
    
    knnreg = KNeighborsRegressor(n_neighbors = K).fit(X_train, y_train)
    
    y_predict_output = knnreg.predict(X_predict_input)
    
    thisaxis.set_xlim([-2.5, 0.75])
    thisaxis.plot(X_predict_input, y_predict_output, '^', markersize = 10,
                 label='Predicted', alpha=0.8)
    thisaxis.plot(X_train, y_train, 'o', label='True Value', alpha=0.8)
    thisaxis.set_xlabel('Input feature', fontsize = 12)
    thisaxis.set_ylabel('Target value', fontsize = 12)
    thisaxis.set_title('KNN regression (K={})'.format(K), fontsize = 12)
    thisaxis.legend()
    
plt.tight_layout()
plt.show()

#==============================================================================
# Linear Regression Learner
#==============================================================================

# Test-Train splitting


X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,
                                                   random_state = 0)
# Linear Regressor Model Fitting

linreg = LinearRegression().fit(X_train, y_train)

print('Linear Model Slope (m): {}'
      .format(linreg.coef_))
print('Linear Model Intercept (b): {:.3f}'
      .format(linreg.intercept_))
print('R-squared score (training): {:.3f}'
      .format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
      .format(linreg.score(X_test, y_test)))

# Plotting the fitted line

plt.figure(figsize=(5,4))

plt.scatter(X_R1, y_R1, marker= 'o', s=50, alpha=0.8)

plt.plot(X_R1, linreg.coef_ * X_R1 + linreg.intercept_, 'r-')
plt.title('Least-squared Linear Regression', fontsize = 12)
plt.xlabel('Feature value (x)',fontsize = 12)
plt.ylabel('Target value (y)', fontsize = 12)
plt.show()

#==============================================================================
# Ridge Regression Learner
#==============================================================================

# Model fitting

linridge = Ridge(alpha=20.0).fit(X_train, y_train)
            # Learning Rate alpha = 20.0

print("Ridge Regression without feature scaling:")            
print('Ridge Regression Linear Model Intercept: {:.3f}'
     .format(linridge.intercept_))
print('Ridge Regression Linear Model Slope:{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))

#Feature scaling
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)
            # Learning Rate alpha = 20.0

print("Ridge Regression after feature scaling:")              
print('Ridge Regression Linear Model Intercept: {:.3f}'
     .format(linridge.intercept_))
print('Ridge Regression Linear Model Slope:{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))

# Effect of varying alpha

print('Ridge regression: Effect of varying alpha regularization parameter:')

for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
    
    linridge = Ridge(alpha = this_alpha).fit(X_train_scaled, y_train)
    r2_train = linridge.score(X_train_scaled, y_train)
    r2_test = linridge.score(X_test_scaled, y_test)
    
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    
    print('Alpha = {:.2f}\n No. of abs(coeff) > 1.0: {}, \
R-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(this_alpha, num_coeff_bigger, r2_train, r2_test))
    
#==============================================================================
# Lasso Regression Learner
#==============================================================================

#Feature scaling

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model fitting
linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)

print('Lasso Regression Linear Model Intercept: {:.3f}'
     .format(linlasso.intercept_))
print('Lasso Regression Linear Slope:{}'
     .format(linlasso.coef_))
print('Non-zero features: {}'
     .format(np.sum(linlasso.coef_ != 0)))
print('R-squared score (training): {:.3f}'
     .format(linlasso.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}\n'
     .format(linlasso.score(X_test_scaled, y_test)))

# Effect of varying alpha

print('Ridge regression: Effect of varying alpha regularization parameter:')

for alpha in [0.5, 1, 2, 3, 5, 10, 20, 50]:
    
    linlasso = Lasso(alpha, max_iter = 10000).fit(X_train_scaled, y_train)
    
    r2_train = linlasso.score(X_train_scaled, y_train)
    r2_test = linlasso.score(X_test_scaled, y_test)
    
    print('Alpha = {:.2f}\nFeatures kept: {}, r-squared training: {:.2f}, \
R-squared test: {:.2f}\n'
         .format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))
#==============================================================================
