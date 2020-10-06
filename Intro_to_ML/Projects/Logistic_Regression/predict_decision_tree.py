# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:24:39 2020

@author: Prithivee Ramalingam
"""

import pandas as pd
import utils
from sklearn import tree, model_selection


train = pd.read_csv("train.csv")
utils.clean_data(train)

target = train["Survived"].values
feature_names = ["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]
features = train[feature_names].values

decision_tree = tree.DecisionTreeClassifier(random_state=1)
decision_tree_ = decision_tree.fit(features,target)
#This causes the model to overfit resulting in a very high score
print(decision_tree_.score(features,target))

#cv = 50 means 50 iterations
scores = model_selection.cross_val_score(decision_tree, features, target, scoring = "accuracy", cv = 50)
print(scores)
print(scores.mean())

generalised_tree = tree.DecisionTreeClassifier(random_state=1, max_depth=7, min_samples_split=2)
generalised_tree_ = generalised_tree.fit(features,target)
#This causes the model to overfit resulting in a very high score
print(generalised_tree_.score(features,target))

#cv = 50 means 50 iterations
scores = model_selection.cross_val_score(generalised_tree, features, target, scoring = "accuracy", cv = 50)
print(scores)
print(scores.mean())

tree.export_graphviz(generalised_tree_, feature_names = feature_names, out_file = "tree.dot")
