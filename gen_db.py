# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:55:39 2018

@author: fnord
"""
from sklearn.model_selection import StratifiedShuffleSplit
from ml_data import SequenceNucsData
import numpy as np


def load_partition(train_index, test_index, X, y):
    x_train = X[train_index,:]
    y_train = y[train_index]    
    x_test = X[test_index,:]
    y_test = y[test_index]    
    return (x_train, y_train), (x_test, y_test)

organism = 'Bacillus'
npath, ppath = './fasta/{}_neg.fa'.format(organism), './fasta/{}_pos.fa'.format(organism)

k=1

data = SequenceNucsData(npath, ppath, k=k)

X = data.getX()
y = data.getY()

kf = StratifiedShuffleSplit(n_splits=5, random_state=34267)
kf.get_n_splits(X, y)

partition = 0 
for train_index, test_index in kf.split(X, y):
    partition += 1
    (x_train, y_train), (x_test, y_test) = load_partition(train_index, test_index, X, y)

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    
    m_train = np.hstack([x_train, y_train])
    m_test = np.hstack([x_test, y_test])
    
    # np.savetxt('./db/{}-p{}-train.nucs'.format(organism, partition), m_train, delimiter=';', fmt='%d')
    # np.savetxt('./db/{}-p{}-test.nucs'.format(organism, partition), m_test, delimiter=';', fmt='%d')
    
    break

test_file = np.loadtxt('./db/{}-p{}-test.nucs'.format(organism, partition), delimiter=';', dtype='int32')

test_values, test_labels = test_file[:, :-1], test_file[:, -1]

print(test_values)
print(test_values.shape[1])
print(test_labels)


print('END')