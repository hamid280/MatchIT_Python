#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 3 16:37:06 2019

@author: mahyar
"""
from dateutil import parser
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

"""
-A Unicode string is a sequence of code points,
which are numbers from 0 through 0x10FFFF (1,114,111 decimal).
This sequence of code points needs to be represented in memory as a set of code units,
and code units are then mapped to 8-bit bytes. The rules for translating a Unicode string
into a sequence of bytes are called a character encoding, or just an encoding.
"""

"""
Exercise 7 - C
"""
data2 = np.genfromtxt('database.csv', delimiter=',')  # To see the default data type
data = np.genfromtxt('database.csv', delimiter=',', dtype=('unicode'))
Depth = data[1:, 5:6].astype(float)  # Convert the string to float
Magnitude = data[1:, 8:9].astype(float)

print("----------------------------------------------")
print(" Exercise 7 - C\n")
print("minimum, maximum and mean Magnitude values:")  # Print the minimum, maximum and mean Magnitude values
print(f" {np.min(Magnitude)}, {np.max(Magnitude)}, {np.mean(Magnitude)}")
print("----------------------------------------------")

"""
Exercise 7 - D
"""
print("----------------------------------------------")
print(" Exercise 7 - D\n")
print("----------------------------------------------")
"""
Calculate the total number of earthquakes for each year and store it inside dictionary 'result'.
"""
"""
3rd way
"""
Date_p = np.genfromtxt('database.csv', converters={0: lambda x: parser.parse(x)}, delimiter=',', usecols=(0),
                       dtype='unicode', skip_header=1)
Type = data[1:, 4:5]

year = []
year_n = []
number = []
for i in Date_p:  # Assign each year to a list
    year_n.append(i.year)

year_n = list(np.sort(year_n))
for j in range(len(year_n)):  # Add each new year to a list and count the number of earthquakes.
    if Type[j][0] == 'Earthquake':
        if year_n[j - 1] != year_n[j]:
            year.append(year_n[j])
number.append(year_n.count(year_n[j]))

print(number)
print(year)

