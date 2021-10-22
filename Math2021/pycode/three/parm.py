#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/17 15:26
@Author :LuneD
@File   :parm.py
"""
import matplotlib
import matplotlib.pyplot as plt
import xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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

#分成7次模型可以
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Choose all predictors except target & IDcols

cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1,
                'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0,
                'reg_lambda': 1}

model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)
#evalute_result = optimized_GBM.
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))