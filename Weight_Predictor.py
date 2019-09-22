# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 06:05:56 2019

@author: Nishant Garg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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

#Avoid Dummy variable trap
X = X[:,1:]


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 0, test_size = 0.2)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
ct_2 = ColumnTransformer(transformers = [("scaler", StandardScaler(),[6,7,8,9,10])], remainder = "passthrough")
X_train = ct_2.fit_transform(X_train)

X_test = ct_2.transform(X_test)


"""Simple Linear Regression"""
#Simple Linear regression
from sklearn.linear_model import LinearRegression
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


"""Multiple Linear Regression"""
#Declaring the co-officient b0
X_train = np.append(arr = np.ones((127,1), dtype=int), values = X_train, axis = 1)
X_test = np.append(arr = np.ones((32,1), dtype=int), values = X_test, axis = 1)

import statsmodels.api as sm
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

