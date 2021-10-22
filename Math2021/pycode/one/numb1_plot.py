#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/14 20:51
@Author :LuneD
@File   :numb1_plot.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

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

# 得到了最优参数，训练随机森林解决回归问题
regressor = RandomForestRegressor(n_estimators=141, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
# 评估回归性能
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

