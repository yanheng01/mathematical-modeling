#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/16 14:28
@Author :LuneD
@File   :number3_frame.py
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

len = dataSet.columns
#len长度为476
featureArray = dataSet.columns[5:476]
names = featureArray
res = dataSet.columns[0]

#训练集
X = dataSet[featureArray]
#测试集
y = dataSet[res]
seed = 7

#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))