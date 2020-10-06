# -*- coding: utf-8 -*-


import pandas as pd
import utils
import matplotlib.pyplot as plt
from sklearn import linear_model

#********************************************************************************************************************
direct = 'V:/DataScience_2019501043/Intro_to_ML/Projects/Logistic_Regression'
test = direct +'/test.csv'
train = direct +'/train.csv'
test = pd.read_csv(test)
train = pd.read_csv(train)
print(test.head(6))

#********************************************************************************************************************
fig = plt.figure(figsize=(18,6))

# (2,3) represents the number of rows and columns we want totally. 2 rows and 3 columns
# The next parameter(0,0) represents that the image should be in 0th row and 0th column
plt.subplot2grid((2,3),(0,0))
# normalise is used to represent the data in percentages
train.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Survived")

plt.subplot2grid((2,3),(0,1))
plt.scatter(train.Survived,train.Age,alpha=0.1)
plt.title("Age with respect to survival")

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

#********************************************************************************************************************
female_color = "#FA0000"
fig = plt.figure(figsize=(18,6))

plt.subplot2grid((3,4),(0,0))
train.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Survived")

plt.subplot2grid((3,4),(0,1))
train.Survived[train.Sex == "male"].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Men Survived")

plt.subplot2grid((3,4),(0,2))
train.Survived[train.Sex == "female"].value_counts(normalize=True).plot(kind="bar",alpha=0.5, color = female_color)
plt.title("Women Survived")

plt.subplot2grid((3,4),(0,3))
train.Sex[train.Survived == 1].value_counts(normalize=True).plot(kind="bar",alpha=0.5, color = [female_color,'b'])
plt.title("Sex of Survival")

plt.subplot2grid((2,3),(1,0),colspan=4)
for x in[1,2,3]:
    train.Survived[train.Pclass == x].plot(kind="kde")
plt.title("Class wrt Survived")
plt.legend(("1st","2nd","3rd"))

plt.subplot2grid((3,4),(2,0))
train.Survived[(train.Sex == "male") & (train.Pclass == 1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Rich men Survived")

plt.subplot2grid((3,4),(2,1))
train.Survived[(train.Sex == "male") & (train.Pclass == 3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Poor men Survived")

plt.subplot2grid((3,4),(2,2))
train.Survived[(train.Sex == "female") & (train.Pclass == 1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5
                                                                                                ,color=female_color)
plt.title("Rich women Survived")

plt.subplot2grid((3,4),(2,3))
train.Survived[(train.Sex == "female") & (train.Pclass == 3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5
                                                                                                ,color=female_color)
plt.title("Poor women Survived")

plt.show()
#********************************************************************************************************************

train.Fare = train.Fare.fillna(train.Fare.dropna().median())
train.Age = train.Age.fillna(train.Age.dropna().median())

# train.loc[train.Sex== "male",Sex] = 0
# train.loc[train.Sex == "female",Sex] = 1
train.loc[train["Sex"] == "male","Sex"] = 0
train.loc[train["Sex"] == "female","Sex"] = 1

train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S","Embarked"] = 0
train.loc[train["Embarked"] == "C","Embarked"] = 1
train.loc[train["Embarked"] == "Q","Embarked"] = 2


train.head(6)
#********************************************************************************************************************

utils.clean_data(train)
train_y = train["Survived"]

