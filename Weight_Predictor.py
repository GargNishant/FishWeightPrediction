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