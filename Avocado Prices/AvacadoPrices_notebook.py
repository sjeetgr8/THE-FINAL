# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 18:58:36 2022

@author: Satya
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import LinearRegression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


os.chdir("E:\Google Drive\Github\THE-FINAL\Avocado Prices")
df= pd.read_csv("avocado.csv")


df.describe()
df=df.drop('Unnamed: 0',axis=1)

df.isna().sum()

df.columns

# remove special character
df.columns = df.columns.str.replace(' ', '')

y=df['AveragePrice']

x=df[['4046','4225','4770','year']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Linear Regression model

# Create linear regression object
regr=linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train,y_train)

# Make predictions using the testing set
y_pred=regr.predict(x_test)

# The coefficients
print("Coefficients: \n",regr.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))

# The coefficient of determination: 1 is perfect 
#predictionprint("Coefficient of determination: %.2f"  % r2_score(diabetes_y_test,diabetes_y_pred))


# Polynomial regression model

features = PolynomialFeatures(degree=4)
x_train_transformed = features.fit_transform(x_train)
model = LinearRegression()
model.fit(x_train_transformed, y_train)


x_test_transformed = features.fit_transform(x_test)

train_pred = model.predict(x_train_transformed)
rmse_poly_4_train = mean_squared_error(y_train, train_pred, squared = False)
print("Train RMSE for Polynonial Regression of degree 4 is {}.".format(rmse_poly_4_train))

#test_pred = model.predict(x_test_transformed)
#rmse_poly_4 = mean_squared_error(y_test, test_pred, squared = False)
#print("Test RMSE for Polynonial Regression of degree 4 is {}.".format(rmse_poly_4))

# Regularization 


reg = linear_model.LassoLars(alpha=.1, normalize=False)
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
