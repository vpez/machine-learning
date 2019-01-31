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

### Feature scaling
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# scale both 
x = scaler_x.fit_transform(x)
y = np.array(y).reshape(-1, 1)
y = scaler_y.fit_transform(y)

# SV regression
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

### Visualizing the Training Set results
### We used feature scaling to fit the model
### To draw the plot using normal values, we have to inverse transform the scaled values
### Both for x and for y

### Datapoints
plt.scatter(scaler_x.inverse_transform(x), scaler_y.inverse_transform(y), color = 'black')

### Smoother line
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.plot(scaler_x.inverse_transform(x_grid), 
         scaler_y.inverse_transform(regressor.predict(x_grid)), 
         color = 'red')

plt.title('Salary vs. Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# values
val_x = scaler_x.transform(np.array([[6.5]]))
val_y = regressor.predict(val_x)
val_y = scaler_y.inverse_transform(val_y)
print('Predicted value using linear model: ' + str(val_y[0]))
