# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:53:21 2020

@author: Prithivee Ramalingam
"""

import pandas as pd
import utils
from sklearn import linear_model, preprocessing

train = pd.read_csv("train.csv")
utils.clean_data(train)

target = train["Survived"].values
features = train[["Pclass","Age","Sex","Fare","Embarked","SibSp","Parch"]].values

classifier = linear_model.LogisticRegression()

classifierLinear = classifier.fit(features,target)
print(classifierLinear.score(features, target))

# In order to fit data which is polynomial, That is power 2. We use preprocessing to fit the data  
poly = preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)

classifierCurved = classifier.fit(poly_features,target)
print(classifierCurved.score(poly_features, target))

