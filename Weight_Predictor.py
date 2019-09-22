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
#Use some function to shuffle the dataset
#Use different Models to predict the weight of Fish including Simple Linear, Multiple, Polynomial Regression

dataset = dataset.sample(frac=1).reset_index(drop=True)
X = dataset.iloc[:,[0,2,3,4,5,6]]
Y = dataset.iloc[:,1]

#Categoring Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct_1 = ColumnTransformer(transformers = [("encoder", OneHotEncoder(),[0])], remainder = "passthrough")
X = np.array(ct_1.fit_transform(X))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 0, test_size = 0.2)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
ct_2 = ColumnTransformer(transformers = [("scaler", StandardScaler(),[7,8,9,10,11])], remainder = "passthrough")
X_train = ct_2.fit_transform(X_train)

X_test = ct_2.transform(X_test)


