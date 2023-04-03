#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:15:16 2023

@author: sakhawathossain
"""

# Step 1: Importing depending lbraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

dataSet = pd.read_excel("/Users/sakhawathossain/Downloads/Happyness.xlsx")

# Step 2: Data cleaning and pre-processing
dataSet = dataSet.dropna() # Here we are removing null values
dataSet = pd.get_dummies(data= dataSet, columns=["What is your gender?(আপনার লিঙ্গ কি?) "])

scaler = StandardScaler()
