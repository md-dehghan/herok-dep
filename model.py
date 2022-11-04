#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:22:22 2022

@author: masoumehdehghani
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# create df
train = pd.read_csv('titanic.csv')

# drop null values
train.dropna(inplace=True)

# features and target
target = 'Survived'
features = ['Pclass', 'Age', 'SibSp', 'Fare']

# X matrix, y vector
X = train[features]
y = train[target]

# model 
regressor = LogisticRegression()
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
