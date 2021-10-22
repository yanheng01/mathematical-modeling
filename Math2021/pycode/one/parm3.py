#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/17 15:12
@Author :LuneD
@File   :parm3.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 导入数据，路径中要么用\\或/或者在路径前加r
dataSet = pd.read_csv(r'D:\csv\md.csv', header=0)

# 准备训练数据
# 提取特征。因变量：1974   729个分子描述符
featureArray = dataSet.columns[1:730]
resArryay = dataSet.columns[0]
X = dataSet[featureArray]
y = dataSet[resArryay]

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 调参，绘制学习曲线来调参n_estimators（对随机森林影响最大）
score_lt = []

# 建立n_estimators为45的随机森林
rfc = RandomForestRegressor(n_estimators=141, random_state=90)

# 用网格搜索调整max_depth
param_grid = {'max_depth': np.arange(1, 20)}
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(X_train, y_train)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)
