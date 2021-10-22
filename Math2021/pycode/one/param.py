#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/17 10:58
@Author :LuneD
@File   :param.py
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

#定义步长
skip = 10

# 每隔10步建立一个随机森林，获得不同n_estimators的得分
for i in range(0, 150, skip):
    rfc = RandomForestRegressor(n_estimators=i+1, random_state=10)
    score = cross_val_score(rfc, X_train, y_train, cv=10).mean()
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)*skip+1))

#最大得分：0.7507955137326452 子树数量为：141

# 绘制学习曲线
colors1 = '#00CED1' #点的颜色
x = np.arange(1,150, skip)
plt.subplot(111)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.plot(x, score_lt, ls='-', lw=1, color=colors1)
plt.scatter(x, score_lt, c=colors1)
plt.title('n_estimators-score-1')
plt.savefig(r'D:\csv\picture\score.png', dpi=300)
plt.show()