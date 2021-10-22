#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/14 20:09
@Author :LuneD
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#训练集
dataSet = pd.read_csv(r'D:\csv\DSM.csv', header=0)

featureArray = dataSet.columns[1:471]
resArryay = dataSet.columns[0]
X = dataSet[featureArray]
y = dataSet[resArryay]
names = featureArray

rf = RandomForestRegressor(n_estimators=141, max_depth=None)

rf.fit(X, y)

importance = rf.feature_importances_

print("Features sorted by their score:")

zip_tmp = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)
#print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))

cnt = 1
res2 = dataSet.columns[0]
list1 = []

list1.append(res2)
for item in range(20):
    list1.append(zip_tmp[item][1])

print(list1)

dataN = dataSet[list1]
print(dataN)

outData = pd.DataFrame(columns=list1, data=dataN)

outData.to_csv('D:\csv\writer.csv', encoding="utf-8")