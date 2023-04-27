# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:45:06 2023

@author: LockedCore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy import stats

def estimate_gaussian(X):
    mu = X.mean(axis = 0)
    sigma = X.var(axis = 0)
    return mu, sigma

def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    
    step = (pval.max() - pval.min()) / 1000
    
    for epsilon in  np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon 
        
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        f1 = (2 * precision * recall) / (precision + recall)
        
        if f1 > best_f1:
            best_epsilon = epsilon
            best_f1 = f1
            
    return best_epsilon, best_f1

def cost(params, Y, R, num_features):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))
    
    J = 0
    
    error = np.multiply((X * theta.T) - Y, R)
    squared_error = np.power(error, 2)
    J = (1 / 2.) * np.sum(squared_error)
    
    return J
    
if __name__ == '__main__':
    
    """
    data = loadmat('data/ex8data1.mat')
    print(data.keys())
    Xval, yval = data['Xval'], data['yval']
    X = data['X']

    fig, ax = plt.subplots(figsize = (12, 8))
    ax.scatter(X[:, 0], X[:, 1])
    plt.show()

    mu, sigma = estimate_gaussian(X)

    xdist = stats.norm(mu[0], sigma[0])
    ydist = stats.norm(mu[1], sigma[1])

    p = np.zeros((X.shape[0], X.shape[1]))
    p[:, 0] = xdist.pdf(X[:, 0])
    p[:, 1] = ydist.pdf(X[:, 1])

    pval = np.zeros((Xval.shape[0], Xval.shape[1]))
    pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])
    pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])

    epsilon = pval.max()
    preds = pval < epsilon 

    epsilon, f1 = select_threshold(pval, yval)

    print(epsilon, f1)

    outliers = np.where(p < epsilon)

    fig, ax = plt.subplots(figsize = (12, 8))
    ax.scatter(X[:, 0], X[:, 1])
    ax.scatter(X[outliers[0], 0], X[outliers[0], 1], color = 'r', s = 50, marker = 'o')
    plt.show()
    """

    data = loadmat('data/ex8_movies.mat')
    Y = data['Y']
    R = data['R']
    print(Y)
    print(R)
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(Y)
    ax.set_xlabel('Users')
    ax.set_ylabel('Movies')
    fig.tight_layout()
    plt.show()
    
    read_params = loadmat('data/ex8_movieParams.mat')
    print(read_params.keys())
    X = read_params['X']
    Theta = read_params['Theta']
    num_users = read_params['num_users'][0][0]
    num_movies = read_params['num_movies'][0][0]
    num_features = read_params['num_features'][0][0]
    
    params =  np.concatenate((np.ravel(X), np.ravel(Theta)))
    print(Y.shape)
    
    print(cost(params, Y, R, num_features))
    
    users = 4
    movies = 5
    features = 3

    X_sub = X[:movies, :features]
    Theta_sub = Theta[:users, :features]
    Y_sub = Y[:movies, :users]
    R_sub = R[:movies, :users]

    params = np.concatenate((np.ravel(X_sub), np.ravel(Theta_sub)))

    print(cost(params, Y_sub, R_sub, features))
    
    