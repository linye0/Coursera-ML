# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 21:09:30 2023

@author: 26294
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

def distance(x, y):
    return np.sum((x - y) ** 2)

def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    closest_centroids = np.zeros(m)
    
    for i in range(m):
        min_dist = float('inf')
        chosen_centroid = -1
        for j in range(k):
            dist = distance(X[i, :], centroids[j, :])
            if dist < min_dist:
                min_dist = dist
                chosen_centroid = j
        closest_centroids[i] = chosen_centroid
    return closest_centroids

def compute_centroids(X, idx, k):
    m, n = X.shape
    computed_centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(idx == i)
        computed_centroids[i, :] = (np.sum(X[indices, :], axis = 1) / len(indices[0])).ravel()
        
    return computed_centroids

if __name__ == '__main__':
    
    data = loadmat('data/ex7data2.mat')

    X = data['X']
    
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    
    closest_centroids = find_closest_centroids(X, initial_centroids)

    print(closest_centroids)
    
    computed_centroids = compute_centroids(X, closest_centroids, 3)
    
    print(computed_centroids)