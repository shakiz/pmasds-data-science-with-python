#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:00:53 2023

@author: sakhawathossain
"""

## REQUIRED LIBRARIES
# For data wrangling 
import numpy as np
import pandas as pd

# For visualization
import matplotlib.pyplot as plt

import seaborn as sns
pd.options.display.max_rows = None
pd.options.display.max_columns = None


# Read the data frame
df = pd.read_csv('/Users/sakhawathossain/Downloads/Churn_Modelling.csv', delimiter=',')
df.shape

# Check columns list and missing values
df.isnull().sum()


# Get unique count for each variable
df.nunique()

# Drop the columns as explained above
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

# Review the top rows of what is left of the data frame
df.head()

# Check variable data types
df.dtypes

labels = 'Exited', 'Retained'
sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.show()

