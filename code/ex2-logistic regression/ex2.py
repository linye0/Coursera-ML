# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:27:40 2023

@author: LockedCore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlt
import scipy.optimize as opt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return 1 / len(X) * np.sum(first - second)

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = 1 / len(X) * np.sum(term)
    
    return grad
    
def predict(theta, X):
    term = X * theta.T
    return np.array([1 if x > 0.5 else 0 for x in term])

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = learningRate / (2 * len(X)) * np.sum(np.power(theta, 2))
    return np.sum(first - second) / len(X) + reg


def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(X[:, i], error)
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = np.sum(term) / len(X) + learningRate / len(X) * theta[:, i]
            
    return grad
        
# regularization

path = 'ex2data2.txt'
data = pd.read_csv(path, header = None, names = ['Test1', 'Test2', 'Pass'])

positive = data[data['Pass'].isin([1])]
negative = data[data['Pass'].isin([0])]

fig, ax = plt.subplots(figsize = (12, 8))
ax.scatter(positive['Test1'], positive['Test2'], s = 50, c = 'b', marker = 'o', label = 'Pass')
ax.scatter(negative['Test1'], negative['Test2'], s = 50, c = 'r', marker = 'x', label = 'Not Pass')
ax.legend()
ax.set_xlabel('Test1')
ax.set_ylabel('Test2')
plt.show()
# 将原始数据转化为多项式

degree = 7
x1 = data['Test1']
x2 = data['Test2']

data.insert(3, 'Ones', 1)
for i in range(1, degree):
    for j in range(i + 1):
        data['F' + str(j) + str(i - j)] = np.power(x1, i - j) + np.power(x2, j)

x1_min, x1_max = min(data['Test1']), max(data['Test1'])
x2_min, x2_max = max(data['Test2']), max(data['Test2'])
data.drop('Test1', axis = 1, inplace = True)
data.drop('Test2', axis = 1, inplace = True)

X = np.matrix(data.iloc[:, 1:])
y = np.matrix(data.iloc[:, 0 : 1])
theta = np.matrix(np.array([0 for _ in range(data.iloc[:, 1:].shape[1])]))
learningRate = 1

print(costReg(theta, X, y, learningRate))
print(gradientReg(theta, X, y, learningRate))

result = opt.fmin_tnc(func = costReg, x0 = theta, fprime = gradientReg, args = (X, y, learningRate))

result_theta = result[0]

print(costReg(result_theta, X, y, learningRate))

"""
# logistic regression
path = 'ex2data1.txt'
data = pd.read_csv(path, header = None, names = ['Exam1', 'Exam2', 'Admitted'])

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize = (12, 8))
ax.scatter(positive['Exam1'], positive['Exam2'], s = 50, c = 'b', marker = 'o', label = 'Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s = 50, c = 'r', marker = 'x', label = 'Rejected')
ax.legend()
ax.set_xlabel('Exam1')
ax.set_ylabel('Exam2')

data.insert(0, 'Ones', 1)
cols = data.shape[1]
rows = data.shape[0]
X = data.iloc[:, 0 : cols - 1]
y = data.iloc[:, cols - 1 : cols]
theta = np.matrix(np.zeros(cols - 1))
print(cost(theta, X, y))
print(gradient(theta, X, y))

result = opt.fmin_tnc(func = cost, x0 = theta, fprime = gradient, args = (X, y))
print(result)

res_theta = result[0]
print(res_theta)

x1 = np.arange(min(data['Exam1']), max(data['Exam1']))
x2 = np.arange(min(data['Exam2']), max(data['Exam2']))
x1, x2 = np.meshgrid(x1, x2)
z = res_theta[0] + res_theta[1] * x1 + res_theta[2] * x2
plt.contour(x1, x2, z, 0)
plt.show()

X = np.matrix(X)
y = np.matrix(y)
res_theta = np.matrix(res_theta)

predicted_label = predict(res_theta, X)
sample_label = np.array(data['Admitted'])

predict_same = [1 if a == b else 0 for (a, b) in zip(predicted_label, sample_label)]
print("The predict accuracy = ", sum(predict_same) / len(predict_same))
"""