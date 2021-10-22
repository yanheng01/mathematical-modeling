#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/16 19:04
@Author :LuneD
@File   :numb2_predict2.py
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
#predict是要预测的题集，有50个样例
dataSet_two = pd.read_csv(r'D:\csv\predict.csv', header=0)

#writer是要训练的集合，有1000+样例
dataSet = pd.read_csv(r'D:\csv\writer.csv', header=0)

featureArray = dataSet.columns[1:21]
resArryay = dataSet.columns[0]
X = dataSet[featureArray]
y = dataSet[resArryay]
names = featureArray

#预测的特征值
X_test = dataSet_two[featureArray]

#使用全部测试集合
rf = RandomForestRegressor(n_estimators=200, max_depth=None)

#拟合
rf.fit(X, y)

#归一化
sc = StandardScaler()
X_test = sc.fit_transform(X_test)

#y_pred是预测值
y_pred = rf.predict(X_test)

print(y_pred)
list_tmp = ['pIC']
outData = pd.DataFrame(columns=list_tmp, data=y_pred)
outData.to_csv('D:\csv\pIC_predict.csv', encoding="utf-8")
