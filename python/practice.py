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

categorical_columns = ['HouseStyle', 'LotShape', 'SaleCondition', 'Utilities']

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
r2 = r2_score(y_test, y_pred)
print()
print("R2 score: " + '{0:.3f}'.format(r2))
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

while (len(include) > 20):
    x_opt = x [:, include]
    regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
    regressor_OLS.summary()
    idx = getMax(regressor_OLS.pvalues)
    del(include[idx])

print(regressor_OLS.summary())


### Visualizing the Training Set results
#x_name = 'GrLivArea'
#plt.scatter(x_train[:,numeric_columns.index(x_name)], y_train, color = 'black')
#plt.plot(x_train, regressor.predict(x_train), color = 'blue')
#plt.title('Price vs. ' + x_name + ' (Training Set)')
#plt.xlabel(x_name)
#plt.ylabel('Price')
#plt.show()