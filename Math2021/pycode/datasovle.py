#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/17 9:31
@Author :LuneD
@File   :datasovle.py
"""

import matplotlib
import matplotlib.pyplot as plt
import xgboost
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# load data
#DSM3中的数据都被处理过了
dataSet = pd.read_csv(r'D:\csv\DSM.csv', header=0)
#predict_Set1中的数据都是训练集
dataSet_two = pd.read_csv(r'D:\csv\predict_set1.csv', header=0)

length = len(dataSet.columns)
length2 = len(dataSet_two.columns)

featureArray = dataSet.columns[1:]
tmp = dataSet_two[featureArray]
res = dataSet.columns[0]

list_tmp=[]

for index in range(50):
    list_tmp.append(list(tmp.loc[index]))

list_Arr = np.array(list_tmp)
list_max = np.max(list_Arr, 0)
list_min = np.min(list_Arr, 0)

list_ag=[]

for index in range(length - 1):
    list_ag.append(list_max[index] - list_min[index])


for i in range(length - 1):
    for j in range(50):
        if list_ag[i] == 0:
            continue
        else: list_tmp[j][i] = (list_tmp[j][i] - list_min[i]) / list_ag[i]

dataN = list_tmp
outData = pd.DataFrame(columns=featureArray, data=dataN)

outData.to_csv(r'D:\csv\data\test.csv', encoding="utf-8")