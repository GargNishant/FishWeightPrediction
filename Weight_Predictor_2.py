#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:09:01 2019

@author: Nishant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Dataset_Fish.csv")


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


one_hot_encoder = ColumnTransformer(transformers = [("encoder",OneHotEncoder(), ["Species"])], remainder = "passthrough")
dataset = one_hot_encoder.fit_transform(dataset)


""" Converting numpy array back to dataframe and adding column names for understanding"""
dataset = pd.DataFrame(dataset)
dataset.columns = ["Bream", "Parkii","Perch", "Pike", "Roach", "Smelt","WhiteFish","Weight" ,"Length1", "Length2", "Length3","Height","Width"]
#Shuffling the DataSet, resetting the Indexes and removing the old Indexes
dataset = dataset.sample(frac = 1).reset_index(drop = True)
X = dataset.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12]].values
Y = dataset.iloc[:,7].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

""" Scaling the Features Length1, Length2, Length3, Width, Height"""
from sklearn.preprocessing import StandardScaler
scaler = ColumnTransformer(transformers = [("scaler",StandardScaler(),[7,8,9,10,11])], remainder = "passthrough")
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

