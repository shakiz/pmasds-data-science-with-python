# -*- coding: utf-8 -*-
"""
Created on Sat May 28 18:04:02 2022

@author: BS846
"""

import numpy as np
from sklearn.linear_model import LinearRegression

dataset1 = np.array([[1,2,3,4,5,6,7,8,9,10], [10,20,30,40,50,60,70,80,90,100]])
dataset2 = np.array([[109,108,107,1067,105,104,103,102,101,121], [119,128,137,167,125,114,133,162,171,181]])

reg = LinearRegression().fit(dataset1, dataset2)



print("------------")
print("Printing Coef")
print(reg.coef_)
print("------------")

print("------------")
print("Printing Intercept")
print(reg.intercept_)
print("------------")