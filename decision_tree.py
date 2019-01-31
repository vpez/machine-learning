#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:56:19 2019

@author: vahepezeshkian
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('datasets/Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)

# Using Random Forest
#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 22, random_state = 0)

regressor.fit(x, y)

y_pred = regressor.predict([[6.5]])

### Visualizing
plt.scatter(x, y, color = 'black')
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.plot(x_grid, regressor.predict(x_grid), color = 'red')
plt.title('Salary vs. Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
