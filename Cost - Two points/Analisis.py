# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:30:56 2019

@author: arodriguez
"""

import pandas as pd
import numpy as np

# read data
data = pd.read_csv("fitness.txt", sep = ",", header = None)
data2 = pd.read_csv("individuos.txt", sep = ",", header = None)
data.columns = ["i", "c", "m", "fitness"]
data2.columns = ["i", "indi"]

# Analyze data
parameters= [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)] 
for c, m in parameters:
    datos = data[data["c"] == c]
    datos = datos[datos["m"] == m]
    print("c = ",c) 
    print("m = ",m) 
    print(min(datos["fitness"]))
    print(np.mean(datos["fitness"]))
    print(np.std(datos["fitness"]))

menor = data[data["fitness"] == min(data["fitness"])]


