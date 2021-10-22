#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/15 15:29
@Author :LuneD
@File   :data_test.py
"""


import matplotlib
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#训练集使用已经筛选出来的20个特征
dataSet = pd.read_csv(r'D:\csv\writer.csv', header=0)
dataSet2 = pd.read_csv(r'D:\csv\predict_set1.csv', header=0)

featureArray = dataSet.columns[1:21]
resArryay = dataSet.columns[0]
X = dataSet[featureArray]
y = dataSet[resArryay]
names = featureArray

dataN = dataSet2[featureArray]

outData = pd.DataFrame(columns=featureArray, data=dataN)

outData.to_csv('D:\csv\predict.csv', encoding="utf-8")

