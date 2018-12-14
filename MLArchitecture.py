# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:29:59 2018

@author: fnord
"""

import numpy as np

class MLArchitecture(object):
    def __init__(self, input_size, args):
        self.input_size = input_size;
        self.architecture = self.define_model_architecture(args)
    
    def define_model_architecture(self, args):
        from keras import layers, models

        maxlen = self.input_size

        x = layers.Input(shape=(maxlen,), dtype='int32', name='input')
        embed = layers.Embedding(5, 9, input_length=maxlen, name='emb')(x)

        conv1 = layers.Conv1D(filters=100, kernel_size=15, strides=1, activation='relu', name='conv1')(embed)
        pool1 = layers.MaxPooling1D(pool_size=2, strides=1, name='pool1')(conv1)

        conv2 = layers.Conv1D(filters=250, kernel_size=17, strides=1, activation='relu', name='conv2')(pool1)
        pool2 = layers.MaxPooling1D(pool_size=2, strides=1, name='pool2')(conv2)

        flat1 = layers.Flatten()(pool2)

        dense1 = layers.Dense(128, activation='relu')(flat1)
        outputs = layers.Dense(1, activation='sigmoid')(dense1)

        architecture = models.Model(inputs=[x], outputs=outputs)
        return architecture
              
    
    def get_architecture(self):
        return self.architecture
        