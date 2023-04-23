# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 21:09:30 2023

@author: 26294
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm
from IPython.display import display, Image
from skimage import io

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

def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    centroids = initial_centroids
    
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
        
    return idx, centroids
        
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)
    
    for i in range(k):
        centroids[i, :] = X[idx[i], :]
    
    return centroids

if __name__ == '__main__':
    
    """
    data = loadmat('data/ex7data2.mat')

    X = data['X']
    
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    
    closest_centroids = find_closest_centroids(X, initial_centroids)
    
    idx, centroids = run_k_means(X, initial_centroids, 1000)
    
    print(centroids)
    
    random_initial_centroids = init_centroids(X, 3)
    
    rndidx, rndcentroids = run_k_means(X, random_initial_centroids, 1000)
    
    print(rndcentroids)
    
    data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
    data2.head()
    
    sb.set(context="notebook", style="white")
    sb.lmplot('X1', 'X2', data=data2, fit_reg=False)
    plt.show()
    
    cluster1 = X[np.where(idx == 0)[0], :]
    cluster2 = X[np.where(idx == 1)[0], :]
    cluster3 = X[np.where(idx == 2)[0], :]
    
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.scatter(cluster1[:, 0], cluster1[:, 1], color = 'r', s = 30, label = 'Cluster1')
    ax.scatter(cluster2[:, 0], cluster2[:, 1], color = 'g', s = 30, label = 'Cluster2')
    ax.scatter(cluster3[:, 0], cluster3[:, 1], color = 'b', s = 30, label = 'Cluster3')
    """
    
    """
    img = Image(filename = 'data/bird_small.png')
    display(img)
    
    image_data = loadmat(file_name = 'data/bird_small.mat')
    A = image_data['A']
    print(A.shape)
    
    A = A / 255
    
    X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
    
    initial_centroids = init_centroids(X, 16)
    
    idx, centroids = run_k_means(X, initial_centroids, 10)
    
    idx = find_closest_centroids(X, centroids)
    
    X_recovered = centroids[idx.astype(int), :]
    
    X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
    
    plt.imshow(X_recovered)
    
    plt.show()
    """