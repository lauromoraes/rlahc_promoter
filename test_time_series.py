# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:40:15 2018

@author: fnord
"""
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from sklearn.datasets import fetch_mldata
from time import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import time
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
# from ggplot import *

from ml_data import *

organism = 'Bacillus'
npath, ppath = './fasta/{}_neg.fa'.format(organism), './fasta/{}_pos.fa'.format(organism)

print(npath, ppath)

k = 1
max_features = 4**k
samples = SequenceDinucProperties(npath, ppath)



X = samples.getX()
print(X.shape)
print('>>>')
tmp =X[:,0,30,0]
print(tmp.shape)
print(tmp)
plt.hist(tmp)




sample = X[7,:,:,:]
sample = sample.reshape(sample.shape[2],sample.shape[1])
print(sample.shape)
print(sample)


diff = list()
for i in range(1,sample.shape[1]):
    val = sample[1,i] - sample[1,i-1]
    diff.append(val)

plt.plot(sample[1,:])
plt.plot(diff)

result = seasonal_decompose(sample[1,:], model='additive', freq=1)
#print(result.trend)
#print(result.seasonal)
#print(result.resid)
#print(result.observed)

result.plot()
print('='*10)