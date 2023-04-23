# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:11:19 2023

@author: LockedCore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def costFunction(X, Y, theta):
    inner = np.power((X * theta.T) - Y, 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescend(X, Y, theta, alpha, iters):
    tmp = np.zeros(theta.shape)
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = X * theta.T - Y
        for j in range(parameters):
            term = np.multiply(X[:, j], error)
            tmp[0, j] = theta[0, j] - alpha / len(X) * np.sum(term)
        theta = tmp
        cost[i] = costFunction(X, Y, theta)
    return theta, cost

def normalEquation(X, Y):
    ret = np.linalg.inv(X.T * X) * X.T * Y
    return ret

"""
# 单变量线性回归
path = 'D:\CSdiy\Coursera-ML\Coursera-ML-AndrewNg-Notes-master\code\ex1-linear regression\ex1data1.txt'
data = pd.read_csv(path, header = None, names = ['Population', 'Profit'])
data.plot(kind = 'scatter', x = 'Population', y = 'Profit', figsize = (12, 8))
plt.show()

data.insert(0, 'Ones', 1)
cols = data.shape[1]
rows = data.shape[0]
X = data.iloc[:, 0 : cols - 1]
Y = data.iloc[:, cols - 1 : cols]
X, Y, theta = np.matrix(X), np.matrix(Y), np.matrix(np.array([0, 0]))
g, cost = gradientDescend(X, Y, theta, 0.01, 1000)
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + g[0, 1] * x 

fig, ax = plt.subplots(figsize = (12, 8))
ax.plot(x, f, 'r', label = 'Prediction')
ax.scatter(data.Population, data.Profit, label = 'Training Data')
ax.legend(loc = 2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted vs. Population size')
plt.show()
"""

"""
# 多变量线性回归
path =  'D:\CSdiy\Coursera-ML\Coursera-ML-AndrewNg-Notes-master\code\ex1-linear regression\ex1data2.txt'
data = pd.read_csv(path, header = None, names = ['Size', 'BedroomNum', 'Price'])
data = (data - data.mean()) / data.std()

data.insert(0, 'Ones', 1)
cols = data.shape[1]
rows = data.shape[0]
X = data.iloc[:, : cols - 1]
Y = data.iloc[:, cols - 1 : cols]
X, Y, theta = np.matrix(X), np.matrix(Y), np.matrix(np.array([0, 0, 0]))
g, cost = gradientDescend(X, Y, theta, 0.01, 1650)
print(g)
"""

"""
# 多元线性回归公式解
path =  'D:\CSdiy\Coursera-ML\Coursera-ML-AndrewNg-Notes-master\code\ex1-linear regression\ex1data2.txt'
data = pd.read_csv(path, header = None, names = ['Size', 'BedroomNum', 'Price'])
data = (data - data.mean()) / data.std()

data.insert(0, 'Ones', 1)
cols = data.shape[1]
rows = data.shape[0]
X = data.iloc[:, : cols - 1]
Y = data.iloc[:, cols - 1 : cols]
X, Y, theta = np.matrix(X), np.matrix(Y), np.matrix(np.array([0, 0, 0]))
theta = normalEquation(X, Y)
print(theta)
"""