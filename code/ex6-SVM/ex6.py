# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:20:06 2023

@author: LockedCore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")

raw_data = loadmat('data/ex6data1.mat')

data = pd.DataFrame(raw_data['X'], columns = ['X1', 'X2'])
data['y'] = pd.DataFrame(raw_data['y'])
positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize = (12, 8))
ax.scatter(positive['X1'], positive['X2'], s = 50, marker = 'x', label = 'Positive')
ax.scatter(negative['X1'], negative['X2'], s = 50, marker = 'o', label = 'Negative')
ax.legend()
plt.show()

svc1 = svm.LinearSVC(C = 1, loss = 'hinge', max_iter = 1000)

svc1.fit(data[['X1', 'X2']], data['y'])
print('C = 1, svc1.score = {}'.format(svc1.score(data[['X1', 'X2']], data['y'])))

data['SVM 1 Confidence'] = svc1.decision_function(data[['X1', 'X2']])

fig, ax = plt.subplots(figsize = (12, 8))
ax.scatter(data['X1'], data['X2'], s = 50, c = data['SVM 1 Confidence'], cmap='seismic')
ax.set_title('SVM(C=1) Decision Confidence')
plt.show()

svc2 = svm.LinearSVC(C = 100, loss = 'hinge', max_iter = 1000)

svc2.fit(data[['X1', 'X2']], data['y'])
print('C = 100, svc2.score = {}'.format(svc2.score(data[['X1', 'X2']], data['y'])))

data['SVM 2 Confidence'] = svc2.decision_function(data[['X1', 'X2']])

fig, ax = plt.subplots(figsize = (12, 8))
ax.scatter(data['X1'], data['X2'], s = 50, c = data['SVM 2 Confidence'] * 100, cmap='seismic')
ax.set_title('SVM(C=100) Decision Confidence')
plt.show()

