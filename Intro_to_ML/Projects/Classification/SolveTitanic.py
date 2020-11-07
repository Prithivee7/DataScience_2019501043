# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 12:29:11 2020

@author: Prithivee Ramalingam
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression #supervised
from sklearn.svm import SVC #unsupervised
from sklearn.neighbors import KNeighborsClassifier #supervised
from sklearn.tree import DecisionTreeClassifier #supervised
from sklearn.ensemble import RandomForestClassifier #supervised

direct = 'V:/DataScience_2019501043/Intro_to_ML/Projects/Classification'
test = direct +'/test.csv'
train = direct +'/train.csv'

test = pd.read_csv(test)
train = pd.read_csv(train)

print(train.info())
print(test.info())

print(train.isnull().sum())
print(test.isnull().sum())


train_test_data = [train, test] # combining train and test dataset
#print("test train size")
#print(train_test_data)
#print(len(train_test_data))

fig = plt.figure(figsize=(18,6))

# (2,3) represents the number of rows and columns we want totally. 2 rows and 3 columns
# The next parameter(0,0) represents that the image should be in 0th row and 0th column
plt.subplot2grid((2,3),(0,0),colspan = 2)
# normalise is used to represent the data in percentages
train.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Survived")

plt.subplot2grid((2,3),(0,2))
train.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Class")

# colspan tells how many columns should the image occupy
# kde - kernel density estimation
plt.subplot2grid((2,3),(1,0),colspan=2)
for x in[1,2,3]:
    train.Age[train.Pclass == x].plot(kind="kde")
plt.title("Class wrt Age")
plt.legend(("1st","2nd","3rd"))

plt.subplot2grid((2,3),(1,2))
train.Embarked.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Embarked")

plt.show()

#print(train["Name"])
for dataset in train_test_data:
    #print(dataset)
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')


s = pd.crosstab(train['Title'], train['Sex'])
#print(s)
#print("***************")


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#print(dataset["Title"])    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
    
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

for dataset in train_test_data:
    #print(dataset.Embarked.unique())
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

#Divide into 5 bins , they will have unequal number of entities
train['AgeBand'] = pd.cut(train['Age'], 5)
print(train['AgeBand'])

for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    
#Divide into 4 bins, Each bin should have equal number of entities
train['FareBand'] = pd.qcut(train['Fare'], 4)
print(train['FareBand'])

for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)

for col in train.columns: 
    print(col)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
for col in test.columns: 
    print(col) 
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1)

k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)

lr = LogisticRegression()
score = cross_val_score(lr, X_train, y_train, cv=k_fold)
#print(score)
print("Logistic Regression = "+str(score.mean()))

svc = SVC()
score = cross_val_score(svc, X_train, y_train, cv=k_fold)
#print(score)
print("Support Vector Classifiers = "+str(score.mean()))


knc = KNeighborsClassifier()
score = cross_val_score(knc, X_train, y_train, cv=k_fold)
#print(score)
print("K Neighbours Classifier = "+str(score.mean()))


d_tree = DecisionTreeClassifier()
score = cross_val_score(d_tree, X_train, y_train, cv=k_fold)
#print(score)
print("Decision Tree = "+str(score.mean()))

#to improve predictive accuracy and prevent overfitting
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
y_pred_random_forest = rfc.predict(X_test)
score = cross_val_score(rfc, X_train, y_train, cv=k_fold)
#print(score)
print("Random Forest Classifier = "+str(score.mean()))
#print(y_pred_random_forest)


#submission = pd.DataFrame({
#        "PassengerId": test["PassengerId"],
#        "Survived": y_pred_random_forest
#    })

#submission.to_csv('submission.csv', index=False)