# -*- coding: utf-8 -*-
"""
Created on Sat May 28 21:06:39 2022

@author: Shakil
"""

import pandas as pd


#Read from CSV
datae=pd.read_csv("C:/Users/BS846/Downloads/SampleCSVFile_11kb.csv",na_values=":", encoding = "ISO-8859-1", on_bad_lines='skip')

datae.describe()
print("--------------")
print("Printing Tail:")
print(datae.tail())

print("Printing Head:")
print(datae.head())
print("--------------")


#Write into csv
path='C:/Users/BS846/Downloads/demo_file.csv'

df = pd.DataFrame({'name': ['Raphael', 'Donatello'],
                   'mask': ['red', 'purple'],
                   'weapon': ['sai', 'bo staff']})
df.to_csv(path, index=False)

