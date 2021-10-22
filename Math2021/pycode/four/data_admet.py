#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/17 16:32
@Author :LuneD
@File   :data_admet.py
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

dataSet = pd.read_csv(r'D:\csv\data\DSM4.csv', header=0)

length = len(dataSet.columns)

featureArray = dataSet.columns[6:]
aDMET = dataSet.columns[0:5]
res = dataSet.columns[5:]

aDMETArray = dataSet[aDMET]
result = dataSet[res]
list1 = []

#print(aDMETArray)
#1974 row

#一共10种组合
for index in range(1974):
    flag1 = True
    flag2 = True
    mark1 = aDMETArray.loc[index][0]    #1
    mark2 = aDMETArray.loc[index][1]    #1
    mark3 = aDMETArray.loc[index][2]    #0
    mark4 = aDMETArray.loc[index][3]    #1
    mark5 = aDMETArray.loc[index][4]    #0
    if mark3 == 0:
        mark3 = 1
        flag1 = False
    if mark5 == 0:
        mark5 = 1
        flag2 = False
    if mark3 == 1 and flag1 == True:
        mark3 = 0
    if mark4 == 1 and flag2 == True:
        mark4 = 0
    if mark1 + mark2 + mark3 + mark4 + mark5 >= 3:
        list1.append(result.loc[index])

dataN = list1
outData = pd.DataFrame(columns=res, data=list1)

outData.to_csv(r'D:\csv\data\ADMET_power.csv', encoding="utf-8")