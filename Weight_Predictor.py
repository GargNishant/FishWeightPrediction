# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 06:05:56 2019

@author: Nishant Garg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def simple_regression(X,Y):
    #Avoid Dummy variable trap
    X_slr = X[:,1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X_slr,Y, random_state = 0, test_size = 0.2)
    #Feature Scaling
    ct_2 = ColumnTransformer(transformers = [("scaler", StandardScaler(),[6,7,8,9,10])], remainder = "passthrough")
    X_train = ct_2.fit_transform(X_train)
    X_test = ct_2.transform(X_test)
    
    slr_result = []
    
    slr = LinearRegression()
    slr.fit(X_train[:,0].reshape(-1,1),Y_train)
    slr_result.append(slr.predict(X_test[:,0].reshape(-1,1)))
    
    slr.fit(X_train[:,1].reshape(-1,1),Y_train)
    slr_result.append(slr.predict(X_test[:,1].reshape(-1,1)))
    
    slr.fit(X_train[:,2].reshape(-1,1),Y_train)
    slr_result.append(slr.predict(X_test[:,2].reshape(-1,1)))
    
    slr.fit(X_train[:,3].reshape(-1,1),Y_train)
    slr_result.append(slr.predict(X_test[:,3].reshape(-1,1)))
    
    slr.fit(X_train[:,4].reshape(-1,1),Y_train)
    slr_result.append(slr.predict(X_test[:,4].reshape(-1,1)))
    
    return Y_test, slr_result


def multiple_regression(X,Y):
    #Avoid Dummy variable trap
    X_mlr = X[:,1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X_mlr,Y, random_state = 0, test_size = 0.2)
    #Feature Scaling
    ct_2 = ColumnTransformer(transformers = [("scaler", StandardScaler(),[6,7,8,9,10])], remainder = "passthrough")
    X_train = ct_2.fit_transform(X_train)
    X_test = ct_2.transform(X_test)
    
    #Declaring the co-officient b0
    X_train = np.append(arr = np.ones((127,1), dtype=int), values = X_train, axis = 1)
    X_test = np.append(arr = np.ones((32,1), dtype=int), values = X_test, axis = 1)
    
    X_train_opt = X_train[:,:]
    
    mlr = sm.OLS(Y_train, X_train_opt).fit()
    mlr.summary()
    

    #Removing x5
    X_train_opt = X_train[:,[0,1,2,3,4,6,7,8,9,10,11]]
    mlr = sm.OLS(Y_train, X_train_opt).fit()
    mlr.summary()
    
    #removing x9
    X_train_opt = X_train[:,[0,1,2,3,6,7,8,9,10]]
    mlr = sm.OLS(Y_train, X_train_opt).fit()
    mlr.summary()
    
    #removing x3
    X_train_opt = X_train[:,[0,1,2,6,7,8,9,10]]
    mlr = sm.OLS(Y_train, X_train_opt).fit()
    mlr.summary()
    
    
    X_train_opt = X_train[:,[0,1,2,6,8,9,10]]
    mlr = sm.OLS(Y_train, X_train_opt).fit()
    mlr.summary()
    
    
    X_train_opt = X_train[:,[0,1,2,6,8,10]]
    mlr = sm.OLS(Y_train, X_train_opt).fit()
    mlr.summary()
    
    
    X_train_opt = X_train[:,[0,2,6,8,10]]
    mlr = sm.OLS(Y_train, X_train_opt).fit()
    mlr.summary()
    
    X_test_opt = X_test[:,[0,2,6,8,10]]
    y_pred_mlr = mlr.predict(X_test_opt)
    return Y_test, y_pred_mlr


def polynomial_regression(X,Y):
    #No Need to take care of dummy vairable trap. The library takes care of it
    poly_reg = PolynomialFeatures(degree = 2)
    X_poly = poly_reg.fit_transform(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_poly,Y,test_size = 0.2, random_state = 0)
    
    lin_reg = LinearRegression()
    lin_reg.fit(X_train,Y_train)
    y_pred_poly = lin_reg.predict(X_test)
    return Y_test, y_pred_poly


dataset = pd.read_csv("Dataset_Fish.csv")
dataset.head()
dataset.describe()


#Use Scaler library to scale the features
#Use Column... library to convert categorical variable "species"
#Avoid Dummy variable trap
#Use some function to shuffle the dataset
#Use different Models to predict the weight of Fish including Simple Linear, Multiple, Polynomial Regression

dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
X = dataset.iloc[:,[0,2,3,4,5,6]]
Y = dataset.iloc[:,1]

#Categoring Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct_1 = ColumnTransformer(transformers = [("encoder", OneHotEncoder(),[0])], remainder = "passthrough")
X = np.array(ct_1.fit_transform(X))


Y_test_slr, slr_result = simple_regression(X,Y)

Y_test_mlr, y_pred_mlr = multiple_regression(X,Y)

Y_test_ply, y_pred_poly = polynomial_regression(X,Y)