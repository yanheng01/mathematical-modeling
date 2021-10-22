#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/17 18:36
@Author :LuneD
@File   :data_dealwith.py
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

dataSet = pd.read_csv(r'D:\csv\writer.csv', header=0)
dataSet2 = pd.read_csv(r'D:\csv\data\ADMET_power.csv', header=0)

#特征的变化
featureArray = dataSet.columns[:]
#pi的值
resArryay = dataSet.columns[0]


res = dataSet2[featureArray]

outData = pd.DataFrame(columns=featureArray, data=res)
outData.to_csv(r'D:\csv\data\data_final.csv', encoding="utf-8")

