# -*- coding: utf-8 -*-
# Data processing template

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('datasets/Data.csv')

### Independent variables are all but the last one
x = dataset.iloc[:, :-1].values

### Dependent variables is the 3rd column
y = dataset.iloc[:, 3].values

### Taking care of missing data
# Note: Using Cmd+i shows the help (documentation) for the function
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

### Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_x = LabelEncoder()

# Fit and transform: Learns the pattern of the datapoints and transforms to numerical values
# Note: The problem of treating country names as numbers: comparison makes no sense
x[:, 0] = labelEncoder_x.fit_transform(x[:, 0])

# Apply label encoder on dependent variable as well
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Introducing dummy variables
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[0])
x = oneHotEncoder.fit_transform(x).toarray()

### Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

### Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)