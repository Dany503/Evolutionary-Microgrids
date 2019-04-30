# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:13:15 2019

@author: arodriguez
"""

import pandas as pd
import numpy as np

# read data
data = pd.read_csv("fitness.txt", sep = ",", header = None)
data2 = pd.read_csv("individuos.txt", sep = ",", header = None)
data.columns = ["i", "eta", "c", "m", "fitness"]
data2.columns = ["i", "indi"]

# Analyze data
parameters= [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)] 
for c, m in parameters:
    datos = data[data["c"] == c]
    datos = datos[datos["m"] == m]
    eta = 0.1
    aux = datos[data["eta"] == eta]
    print("c = ",c) 
    print("m = ",m) 
    print("eta = ",eta) 
    print(min(aux["fitness"]))
    print(np.mean(aux["fitness"]))
    print(np.std(aux["fitness"]))

#menor = data[data["fitness"] == min(data["fitness"])]
#print(menor)
