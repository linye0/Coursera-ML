# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 23:24:46 2023

@author: LockedCore
"""

import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_data():
    d = sio.loadmat('ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])

def cost(theta, X, y):
    m = X.shape[0]
    
    inner = X @ theta - y
    
    square_sum = inner.T @ inner
    ret = square_sum / (2 * m)
    
    return ret

def regularized_cost(theta, X, y, learningRate):
    m = X.shape[0]
    
    inner = X @ theta - y
    
    square_sum = inner.T @ inner
    ret = square_sum / (2 * m)
    
    ret += np.sum(np.power(theta[1:], 2)) * learningRate / (2 * m)
    
    return ret

def gradient(theta, X, y):
    m = X.shape[0]
    
    inner = X.T @ (X @ theta - y) 
    
    return inner / m

def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = theta.copy()  
    regularized_term[0] = 0  

    regularized_term = (l / m) * regularized_term

    return gradient(theta, X, y) + regularized_term

def linear_regression_np(X, y, l=1):
    """linear regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    theta = np.ones(X.shape[1])

    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    return res

def normalize_feature(df):
    return df.apply(lambda col: (col - col.mean()) / col.std())

def prepare_poly_data(*args, power):
    
    def prepare(x):
        df = poly_features(x, power = power)
        
        ndarr = normalize_feature(df).values
        
        return np.insert(ndarr, 0, values = np.ones(ndarr.shape[0]), axis = 1)
    
    return [prepare(x) for x in args]
        
def poly_features(x, power, as_ndarray = False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    
    df = pd.DataFrame(data)
    
    return df.values if as_ndarray else df

def plot_learning_curve(X, y, Xval, yval, l = 0):
    training_cost, cv_cost = [], []
    m = X.shape[0]
    
    for i in range(1, m + 1):
        res = linear_regression_np(X[:i, :], y[:i], l = l)
        
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)
        
        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label = 'training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label = 'cv cost')
    plt.legend(loc = 1)
    plt.title("l = {}".format(l))
    plt.show()

if __name__ == '__main__':
    X, y, Xval, yval, Xtest, ytest = read_data()
    
    df = pd.DataFrame({'water_level': X, 'flow': y})
    
    X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, values = np.ones(x.shape[0]), axis = 1) for x in (X, Xval, Xtest)]
    
    theta = np.ones(X.shape[0])

    final_theta = linear_regression_np(X, y, l=0).get('x')

    b = final_theta[0] # intercept
    m = final_theta[1] # slope

    plt.scatter(X[:,1], y, label="Training data")
    plt.plot(X[:, 1], X[:, 1]*m + b, label="Prediction")
    plt.legend(loc=2)
    plt.show()
    
    # 判断欠拟合还是过拟合的方法：以训练数据数为横轴，分别画出训练集和验证集上的cost，如果训练集的cost随着训练样本量的增加而增加，说明模型欠拟合
    # 如果训练集的cost减少，而验证集的cost增加，说明过拟合
    
    training_cost, cv_cost = [], []
    
    m = X.shape[0]
    for i in range(1, m + 1):
        theta = linear_regression_np(X[:i, :], y[:i]).get('x')
        tc = cost(theta, X[:i, :], y[:i])
        cv = cost(theta, Xval, yval)
        
        training_cost.append(tc)
        cv_cost.append(cv)
        
    plt.plot(np.arange(1, m+1), training_cost, label='training cost')
    plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
    plt.legend(loc=1)
    plt.show()
    
    X, y, Xval, yval, Xtest, ytest = read_data()
    
    X, Xval, Xtest = prepare_poly_data(X, Xval, Xtest, power = 3)
    
    theta = linear_regression_np(X, y, l = 1).x
    
    x0 = np.linspace(min(X[:, 1]), max(X[:, 1]))
    
    y0 = theta[0] + theta[1] * x0 + theta[2] * x0 ** 2 + theta[3] * x0 ** 3
    plt.scatter(X[:,1], y, label="Training data")
    plt.plot(x0, y0, label="Prediction")
    plt.legend(loc=2)
    plt.show()
    
    # 找到最佳的learningRate
    
    plot_learning_curve(X, y, Xval, yval, l = 0)
    
    l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    training_cost, cv_cost = [], []
    
    for l in l_candidate:
        res = linear_regression_np(X, y, l = l)
        
        theta = res.get('x')
        
        tc = cost(theta, X, y)
        cv = cost(theta, Xval, yval)
        
        training_cost.append(tc)
        cv_cost.append(cv)
        
    plt.plot(l_candidate, training_cost, label = 'Train')
    plt.plot(l_candidate, cv_cost, label = 'Cross Validation')
    plt.legend(loc = 2)
    
    plt.xlabel('lambda')
    
    plt.ylabel('cost')
    
    plt.show()
    
    l_best = l_candidate[np.argmin(cv_cost)]
    
    print(l_best)
    
    # use test data to compute the cost
    for l in l_candidate:
        theta = linear_regression_np(X, y, l).x
        print('test cost(l={}) = {}'.format(l, cost(theta, Xtest, ytest)))