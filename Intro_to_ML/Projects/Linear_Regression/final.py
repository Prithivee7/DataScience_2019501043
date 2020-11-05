# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:08:14 2020

@author: Prithivee Ramalingam
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_squared_error

#19 days , 12 months for 2 years
#11 days, 12 months for 2 years - for every hour find the number of rentals
def load_data_sets():
    train=pd.read_csv('V:/DataScience_2019501043/Intro_to_ML/Projects/Linear_Regression/train.csv')
    test= pd.read_csv('V:/DataScience_2019501043/Intro_to_ML/Projects/Linear_Regression/test.csv')
    return train,test

def explore(train,test):
    print(train.head)
    print(test.head)

    print(train.info())
    print(test.info())

    print(train.isnull().sum())
    print(test.isnull().sum())

def encoding(dataframe):
    print(dataframe['season'].head)
    dataframe['season'] = dataframe['season'].map({1:'Spring',2:'Summer',3:'Fall',4:'Winter'})
    dataframe['holiday'] = dataframe['holiday'].map({0:'Nholiday',1:'Holiday'})
    dataframe['workingday'] = dataframe['workingday'].map({0:'Off',1:'Workday'})
    print(dataframe['season'].head)
    return dataframe

def visualisation(train):
    train_sample = encoding(train)
    
    train_sample.groupby('season')['count'].sum().plot.bar()
    train_sample.groupby('holiday')['count'].sum().plot.bar()
    train_sample.groupby('workingday')['count'].sum().plot.bar()
    
def convert_date_time(dataframe):
    #Converting the attribute datetime from object to datetime type
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])
    dataframe['year'] = dataframe['datetime'].dt.year
    dataframe['month'] = dataframe['datetime'].dt.month
    dataframe['hour'] = dataframe['datetime'].dt.hour
    dataframe['DayOfWeek'] = dataframe['datetime'].dt.dayofweek
    return dataframe
    
def feature_selection(train,test):
    x_train = train[['weather','temp',
                                 'atemp','humidity','windspeed','year','month','hour','DayOfWeek']]
    y_train = train['count']

    x_test = test[['weather','temp','atemp',
                   'humidity','windspeed','year','month','hour','DayOfWeek']]
    return x_train,y_train,x_test

def train_valid_splitting(x_train, y_train):
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.35, random_state = 42)
    return x_train, x_validate, y_train, y_validate

def linear_regression(x_train, y_train, x_validate,y_validate):
    lr = LinearRegression().fit(x_train,y_train)       
    y_predicted_value = lr.predict(x_validate)
    print('r2_score for linear regression:',r2_score(y_validate,y_predicted_value))
    print('rmse for linear regression:',np.sqrt(mean_squared_error(y_validate,y_predicted_value)))     

def decision_tree_regressor(x_train, y_train, x_validate,y_validate):
    dt = DecisionTreeRegressor().fit(x_train,y_train)
    y_predicted_value = dt.predict(x_validate)
    print('r2_score for decision tree regressor:',r2_score(y_validate,y_predicted_value))
    print('rmse for decision tree regressor:',np.sqrt(mean_squared_error(y_validate,y_predicted_value))) 

def random_forest_regressor(x_train, y_train, x_validate,y_validate):
    rf = RandomForestRegressor().fit(x_train,y_train)
    y_predicted_value = rf.predict(x_validate)
    print('r2_score for random forest regressor:',r2_score(y_validate,y_predicted_value))
    print('rmse for random forest regressor:',np.sqrt(mean_squared_error(y_validate,y_predicted_value)))
    return y_predicted_value
         
def ridge_regressor(x_train, y_train, x_validate, y_validate):
    rr = Ridge().fit(x_train,y_train)
    y_predicted_value = rr.predict(x_validate)
    print('r2_score for ridge regression:',r2_score(y_validate,y_predicted_value))
    print('rmse for ridge regression:',np.sqrt(mean_squared_error(y_validate,y_predicted_value)))

def submit(y_predicted_value, test):
    #sample_submission = pd.DataFrame({'datetime':x_test['datetime'],'count':y_predicted_value})
    #sample_submission.to_csv('final2.csv',index=False)
    
    sample_submission = pd.read_csv('sampleSubmission.csv')
    #predicted_count_RFR = RandomForestRegressor.predict(X_test)
    sample_submission['count'] = pd.Series(y_predicted_value.clip(0))
    sample_submission.to_csv('Output.csv', index = False)

train, test = load_data_sets()
explore(train,test)
visualisation(train)
visualisation(test)
train = convert_date_time(train)
test = convert_date_time(test)

#root mean squared error. The smaller the better
#r squared tells how close the data is to the regression line
x_train,y_train,x_test = feature_selection(train,test)
x_train, x_validate, y_train, y_validate = train_valid_splitting(x_train,y_train)
print(x_train.info())
linear_regression(x_train, y_train, x_validate, y_validate)
decision_tree_regressor(x_train, y_train, x_validate,y_validate)
y_predicted_value = random_forest_regressor(x_train, y_train, x_validate,y_validate)
ridge_regressor(x_train, y_train, x_validate, y_validate)

#submit(y_predicted_value,test)
