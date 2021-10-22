#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time   :2021 2021/10/17 21:28
@Author :LuneD
@File   :numb4_code.py
"""

import random
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False

#方程组的系数
lv = [3.362, -0.026, 0.163, -0.52, 1.449, 1.708, 0.182, 0.589, -0.066, 1.506,
              0.022, 0.001, 0.003, 0.086, 0.257, -3.121, -0.025, -0.256, 0.001, -0.278, -0.001]



#定义粒子群算法类PSO
class PSO(object):
    def __init__(self):
        self.x_bound = [-10, 10]    #指定x的范围
        self.T = 100                #迭代的次数
        self.w = 0.15               #公式固定参数
        self.N = 1000               #每次迭代粒子总数
        self.dim = 20               #粒子维度
        self.c1 = 1.5               #公式固定参数
        self.c2 = 1.5               #公式固定参数

#定义y和x的函数函数关系
    def fun(self, x):
        result = lv[0]
        cnt = 0
        for i in x:
            cnt+=1
            if cnt % 2 == 1:
                result += pow(i, 2) * lv[cnt]
            if cnt % 2 == 0:
                result += i * lv[cnt]
        return result

#主函数
    def pso_main(self):
        x = []
        v = []
        for j in range(self.N):
            x.append([random.random() for i in range(self.dim)])
            v.append([random.random() for m in range(self.dim)])
        fitness = [self.fun(x[j]) for j in range(self.N)]
        p = x
        best = max(fitness)
        pg = x[fitness.index(max(fitness))]
        best_all = []
        for t in range(self.T):
            for j in range(self.N):
                for m in range(self.dim):
                    v[j][m] = self.w * v[j][m] + self.c1 * random.random() * (
                            p[j][m] - x[j][m]) + self.c2 * random.random() * (pg[m] - x[j][m])
            for j in range(self.N):
                for m in range(self.dim):
                    x[j][m] = x[j][m] + v[j][m]
                    if x[j][m] > self.x_bound[1]:
                        x[j][m] = self.x_bound[1]
                    if x[j][m] < self.x_bound[0]:
                        x[j][m] = self.x_bound[0]
            fitness_ = []
            for j in range(self.N):
                fitness_.append(self.fun(x[j]))

            #比较，取每次迭代的最大值
            if max(fitness_) > best:
                pg = x[fitness_.index(max(fitness_))]
                best = max(fitness_)
            best_all.append(best)
            print('第' + str(t) + '次迭代：最优解位置在' + str(pg) + '，最优解的适应度值为：' + str(best))
        plt.plot([t for t in range(self.T)], best_all)
        plt.ylabel('适应度值')
        plt.xlabel('迭代次数')
        plt.title('粒子群适应度趋势')
        plt.show()

#调用类
p1 = PSO()
p1.pso_main()
