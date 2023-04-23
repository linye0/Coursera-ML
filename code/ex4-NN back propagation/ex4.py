# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:08:56 2023

@author: LockedCore
"""

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
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def forward_propagate(X, theta1, theta2):
    a1 = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
    z2 = a1 * theta1.T
    
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, np.ones(a2.shape[0]), axis = 1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    # a3.shape == (5000, 10)
    
    return a1, z2, a2, z3, h

def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    J = 0
    
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    
    J = J / m
    
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    
    return J

def backprop(params, input_size, hidden_size, num_labels, X, y, learningRate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, hidden_size + 1)))
    
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    
    delta1 = np.zeros(theta1.shape) # (25, 401)
    delta2 = np.zeros(theta2.shape) # (10, 26)
    
    J = 0
    
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    
    J = J / m
        
    J += (float(learningRate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    
    for t in range(m):
        a1t = a1[t, :] # (1, 401)
        z2t = z2[t, :] # (1, 25)
        a2t = a2[t, :] # (1, 26)
        ht = h[t, :] # (1, 10)
        yt = y[t, :] # (1, 10)
        
        d3t = ht - yt # (1, 10)
        
        z2t = np.insert(z2t, 0, values = np.ones(1)) # (1, 26)
        d2t = np.multiply(d3t * theta2, sigmoid_gradient(z2t))
        
        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
        
    delta1 = delta1 / m
    delta2 = delta2 / m
    
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learningRate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learningRate) / m
    
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
        
    return J, grad

if __name__ == '__main__':
    
    data = loadmat('ex4data1.mat')
    X = data['X']
    y = data['y']
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)
    
    weight = loadmat('ex4weights.mat')
    theta1 = weight['Theta1']
    theta2 = weight['Theta2']
    
    input_size = 400
    hidden_size = 25
    num_labels = 10
    learningRate = 1

    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
    
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # 将参数数组解开为每个层的参数矩阵
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    fmin = opt.minimize(fun = backprop, x0 = params, args = (input_size, hidden_size, num_labels, X, y_onehot, learningRate), method = 'TNC', jac = True, options = {'maxiter': 250})
    
    print(fmin)
    