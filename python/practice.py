#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:30:36 2019

@author: vahepezeshkian
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.formula.api as sm

dataset = pd.read_csv('datasets/housing-prices/train.csv')

numeric_columns = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
                   'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                   '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
                   'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
                   'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                   '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']

categorical_columns = ['HouseStyle', 'LotShape', 'SaleCondition', 'Utilities',
                       'MSZoning', 'Street', 'LotConfig', 'Neighborhood',
                       'SaleType']

x = dataset.iloc[:, 0:80].values
y = np.array(dataset.iloc[:, -1].values).reshape(-1, 1)

### Firstly include the numeric variables
columns = []
for col in numeric_columns:
    columns.append(dataset.columns.get_loc(col))

x = x[:, columns]

### Taking care of missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x)
x = imputer.transform(x)

### Adding categorical variables
categorical_indice = []
for col in categorical_columns:
    categorical_indice.append(dataset.columns.get_loc(col))

 ### Append cat_columns to x
for index in categorical_indice:
    labelEncoder = LabelEncoder()
    
    ### Get the categorical column, encode it and add it to the X
    x = np.append(x, 
                  np.array(labelEncoder.fit_transform(dataset.iloc[:, index].values)).reshape(-1, 1), 
                  axis = 1)
    
    ### Encode the newly added column (it is at the end) as dummy variables
    oneHotEncoder = OneHotEncoder(categorical_features = [x.shape[1] - 1])
    x = oneHotEncoder.fit_transform(x).toarray()
    
    ### Remove first column (to avoid dummy variable trap)
    x = x[:, 1:]

### Scaling
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

### Splitting into TRAINING and TESTING sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

### Fitting the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

### Predict
y_pred = regressor.predict(x_test)

### Accuracy score
r2_before = r2_score(y_test, y_pred)
print()
print("R2 score (Before): " + '{0:.3f}'.format(r2_before))
print()

### Backward elimination

### Maximum index getter
def getMax(p_values):
    index_max = 0
    for i in range(0, len(p_values)):
        if (p_values[i] >= p_values[index_max]):
            index_max = i
    return index_max

### Adding X0
x = np.append(arr = np.ones((x.shape[0], 1)).astype(int), values = x, axis = 1)

include = np.arange(x.shape[1]).tolist()

### Eliminate columns until 10 remain
### Instead, can check the p-value until no p-value is larger than 0.05
while (len(include) > 10):
    x_opt = x [:, include]
    regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
    idx = getMax(regressor_OLS.pvalues)
    del(include[idx])

x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size = 1/3, random_state = 0)

### Fitting the model with polynomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
x_poly = poly.fit_transform(x_train)
regressor_2 = LinearRegression()
regressor_2.fit(x_poly, y_train)

### Predict
y_pred = regressor_2.predict(poly.transform(x_test))

### Accuracy score
r2_after = r2_score(y_test, y_pred)
print()
print("R2 score (After): " + '{0:.3f}'.format(r2_after))
print()

### Plot real values with predictions
size = 30
y_test_val = scaler_y.inverse_transform(y_test[:size, :])
y_pred_val = scaler_y.inverse_transform(y_pred[:size, :])
x_axis = np.arange(size)
plt.scatter(x_axis, y_test_val, color = 'black')
plt.scatter(x_axis, y_pred_val, color = 'red')
plt.show()
