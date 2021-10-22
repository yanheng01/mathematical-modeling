#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/16 21:38
@Author :LuneD
@File   :num_5.py
"""



import matplotlib
import matplotlib.pyplot as plt
import xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# load data
dataSet = pd.read_csv(r'D:\csv\DSM3.csv', header=0)
dS = pd.read_csv(r'D:\csv\predict_set1.csv', header=0)

len = dataSet.columns
#len长度为476
featureArray = dataSet.columns[5:476]
names = featureArray
res_1 = dataSet.columns[4]

#训练集
X = dataSet[featureArray]
#测试集
y = dataSet[res_1]

#测试集里的
X_test = dS[featureArray]

seed = 7

model = XGBClassifier()
model.fit(X, y)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = list(y_pred)
print(predictions)
list1 = ['MN分类结果']

dataN = y_pred
outData = pd.DataFrame(columns=list1, data=dataN)

outData.to_csv('D:\csv\classfication\MN.csv', encoding="utf-8")