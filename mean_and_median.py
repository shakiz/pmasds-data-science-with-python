# -*- coding: utf-8 -*-
"""
Created on Sat May 28 18:10:40 2022

@author: Shakil
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import statistics as stat
from fractions import Fraction as F

x = np.array([[1,1],[1,2],[2,2],[2,3]])
y = np.array([6,8,9,11])



print("------------")
print("Printing Mean:")
print(stat.mean([F(3, 7), F(1, 21), F(5, 3), F(1, 3)]))
print("Printing Median:")
print(stat.median([F(3, 7), F(1, 21), F(5, 3), F(1, 3)]))
print("Printing Median Low:")
print(stat.median_low([F(3, 7), F(1, 21), F(5, 3), F(1, 3)]))
print("Printing Median High:")
print(stat.median_high([F(3, 7), F(1, 21), F(5, 3), F(1, 3)]))
print("Printing Media for Grouped data:")
print(stat.median_grouped([F(3, 7), F(1, 21), F(5, 3), F(1, 3)]))
print("------------")