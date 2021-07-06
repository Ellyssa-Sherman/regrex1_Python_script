#!/usr/bin/env python
# coding: utf-8

#Importing packages to use
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance
import sys
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read in file with argument
dataset = pd.read_csv(sys.argv[1])
print("loading {}".format(sys.argv[1]))


dataset.plot(x='x', y='y', style='.')  
plt.title('Example Linear Model')  
plt.xlabel('X')  
plt.ylabel('Y')  
plt.savefig("Python_Data Scatter.png")


x = dataset['x'].values.reshape(-1,1)
y = dataset['y'].values.reshape(-1,1)
x_train = x
y_train = y
regressor = LinearRegression()  
regressor.fit(x_train, y_train) #training the algorithm
y_pred = regressor.predict(x)
plt.scatter(x, y,  color='black', marker='.')
plt.plot(x, y_pred, color='black', linewidth=1)
plt.savefig("Python_Data Regression.png")