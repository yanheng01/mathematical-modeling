#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/17 17:16
@Author :LuneD
@File   :number1_code.py
"""

#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/14 20:09
@Author :LuneD
@File   :numb1_frame.py
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

list1=[]
list3=[]
list4=[]
for item in range(20):
    list1.append(zip_tmp[item])
    list3.append(list1[item][0])
    list4.append(list1[item][1])
print(list1)
print(list3)
print(range(len(list3)))

list2 = ['Power', 'Name']

plt.title("Feature Importance")

#指定x坐标轴，高度等参数
plt.bar(x=range(20), height=list3, width=0.3, color='lightblue', align='center')
#对x坐标轴的标签进行覆盖,rotation是角度
plt.xticks(range(len(list3)), list4, rotation='vertical')
#限制x轴数据的大小
plt.xlim([-1, len(list1)])
plt.tight_layout()
plt.show()

outData = pd.DataFrame(columns=list2, data=list1)

outData.to_csv('D:\csv\power.csv', encoding="utf-8")