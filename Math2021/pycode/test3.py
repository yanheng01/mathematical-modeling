#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/15 15:40
@Author :LuneD
@File   :test3.py
"""

zip_tmp = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd'], [5, 'e'], [6, 'f']]
list1 = []
for item in range(5):
    list1.append(zip_tmp[item][1])

print(list1)