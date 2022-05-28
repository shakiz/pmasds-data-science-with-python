# -*- coding: utf-8 -*-
"""
Created on Sat May 28 17:50:30 2022

@author: Shakil
"""



import pandas as pd
import numpy as np

data = {"year ": [2010,2011,2012,2010,2011,2012,2010,2011,2012],
"team ": ["FCBarcelona ", "FCBarcelona ","FCBarcelona ", "RMadrid ",
"RMadrid ", "RMadrid ","ValenciaCF ", "ValenciaCF ","ValenciaCF "],
"wins ": [30, 28, 32, 29, 32, 26, 21, 17, 19],
"draws ": [6,7,4,5,4,7,8,10,8],
"losses ": [2, 3, 2, 4, 2, 5, 9, 11, 11]
}

football = pd.DataFrame(data,columns=["year ","team ","wins ","draws ","losses "])
print("------------")
print("All football items are printing:")
print(football)
print("------------")
wina=football["wins "] #selecting data
wina

#filter those values of wins greater than 20
football[football["wins "] > 20].tail()

#sqrt function from the NumPy for square root of each value in the Value column.
s = football["wins "]. apply(np.sqrt)
s.head ()


ss = football["wins "]. apply(lambda d: d**2)
ss.head ()

#sort values 
football.sort_values(by = "wins ")

football.sort_values(by = "wins ",ascending = False)

#using the isnull() function to filter rows with missing values
football[football["wins "]. isnull ()]. head ()
print("------------")
print("Printing Describe:")
print(football.describe())
print("------------")


print("------------")
print("Printing Skew:")
print(football.skew())
print("------------")

print("------------")
print("Printing Head:")
print(football.head())
print("------------")


print("------------")
print("Printing Tail:")
print(football.tail())
print("------------")

