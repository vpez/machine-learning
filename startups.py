#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:41:52 2019

@author: vahepezeshkian
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('datasets/50_Startups.csv')

### Independent variables are all but the last one
x = dataset.iloc[:, :-1].values

### Dependent variables is the 3rd column
y = dataset.iloc[:, -1].values

### Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_x = LabelEncoder()

# Fit and transform: Learns the pattern of the datapoints and transforms to numerical values
# Note: The problem of treating country names as numbers: comparison makes no sense
x[:, 3] = labelEncoder_x.fit_transform(x[:, 3])

# Introducing dummy variables
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[3])
x = oneHotEncoder.fit_transform(x).toarray()

### Avoid the "Dummy variable trap": Remove a column that is correlated
x = x[:, 1:]

### Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

### Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

### Predicting the Test Set results
y_pred = regressor.predict(x_test)



### Backward elimination: Throwing away unnecessary variables
#import statsmodels.formula.api as sm
#x = np.append(values = np.ones((50, 1)).astype(int), arr = x, axis = 1)

#x_opt = x[:, [0, 1, 2, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_OLS.summary()

#x_opt = x[:, [0, 1, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_OLS.summary()

#x_opt = x[:, [0, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_OLS.summary()