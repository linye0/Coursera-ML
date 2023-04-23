# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:54:38 2023

@author: LockedCore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

def gradient(theta, X, y, learningRate):
   theta = np.matrix(theta)
   X = np.matrix(X)
   y = np.matrix(y)
   
   error = sigmoid(X * theta.T) - y
   
   grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
   
   grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
   
   return np.array(grad).ravel()

# 用二分类实现多分类：训练每个label的参数，最后当输入一个图像时，分别计算对应每一个label的概率，输出概率最大的类型
# 训练参数函数
def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    
    all_theta = np.zeros((num_labels, params + 1))
    
    X = np.insert(X, 0, values = np.ones(rows), axis = 1)
    
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        
        fmin = opt.minimize(fun = cost, x0 = theta, args = (X, y_i, learning_rate), method = 'TNC', jac = gradient)
        all_theta[i - 1, :] = fmin.x
        
    return all_theta

def predict_all(X, all_theta):
    rows = X.shape[0]
    
    X = np.insert(X, 0, values = np.ones(rows), axis = 1)
    
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    
    h = sigmoid(X * all_theta.T)
    
    h_argmax = np.argmax(h, axis = 1)
    
    h_argmax = h_argmax + 1
    
    return h_argmax
    
def predict_all_NN(X, theta1, theta2):
    rows = X.shape[0]
    
    X = np.insert(X, 0, values = np.ones(rows), axis = 1)
    
    X = np.matrix(X)
    
    z_2 = X * theta1.T
    a_2 = sigmoid(z_2)
    
    a_2 = np.insert(a_2, 0, values = np.ones(a_2.shape[0]), axis = 1)
    
    z_3 = a_2 * theta2.T
    h_argmax = np.argmax(z_3, axis = 1)
    h_argmax = h_argmax + 1
    return h_argmax



if __name__ == '__main__':
    
    """
    data = loadmat('ex3data1.mat')
    
    all_theta = one_vs_all(data['X'], data['y'], 10, 1)
    
    y_pred = predict_all(data['X'], all_theta)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
    accuracy = (sum(map(int, correct))) / float(len(correct))
    
    print('accuracy = {0}%'.format(accuracy * 100))
    """
    
    data = loadmat('ex3data1.mat')
    X = data['X']
    y = data['y']
    data = loadmat('ex3weights.mat')
    theta1 = data['Theta1']
    theta2 = data['Theta2']
    y_pred = predict_all_NN(X, theta1, theta2)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct))) / float(len(correct))
    print('accuracy = {0}%'.format(accuracy * 100))