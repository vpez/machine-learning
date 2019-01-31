#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:16:58 2019

@author: vahepezeshkian
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('datasets/Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# split? no. the dataset is too small!

# linear regression
from sklearn.linear_model import LinearRegression
regressor_1 = LinearRegression()
regressor_1.fit(x, y)

# polynomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 3)
x_poly = poly.fit_transform(x)

regressor_2 = LinearRegression()
regressor_2.fit(x_poly, y)

### Visualizing the Training Set results
plt.scatter(x, y, color = 'black')
plt.plot(x, regressor_1.predict(x), color = 'blue')

### Smoother line
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.plot(x_grid, regressor_2.predict(poly.transform(x_grid)), color = 'red')

plt.title('Salary vs. Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# values
val_1 = regressor_1.predict([[6.5]])[0]
val_2 = regressor_2.predict(poly.transform([[6.5]]))[0]

print('Predicted value using linear model: ' + str(val_1))
print('Predicted value using polynomial model: ' + str(val_2))