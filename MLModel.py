# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:08:40 2018

@author: fnord
"""

from MLArchitecture import MLArchitecture
import numpy as np
from keras import backend as K
K.set_learning_phase(0)

LOSS_TYPES = ('binary_crossentropy',)
OPTIMIZER_TYPES = ('Adam','Nadam','SGD','RMSprop',)

class MLModel(object):
    def __init__(self, args):
        self.model_cnt = 0
        self.args = args
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.get_input_data()
        self.gen_architecture()
        self.define_calls()

    def eval(self, mask=None):
        del self.model
        K.clear_session()
        self.get_input_data(mask)
        self.define_calls()
        self.gen_architecture()
        self.train_model()
        stats, y_pred = self.test_model()
        return stats.Mcc


    def gen_architecture(self):
        self.architecture = MLArchitecture(self.input_size, self.args)
        self.model = self.architecture.get_architecture()
        self.model.summary()
        return self.model

    def train_model(self):
        from keras import callbacks as C
        from sklearn.model_selection import StratifiedShuffleSplit

        self.compile()

        kf = StratifiedShuffleSplit(n_splits=1, random_state=13, test_size=0.1)
        kf.get_n_splits(self.x_train, self.y_train)

        for t_index, v_index in kf.split(self.x_train, self.y_train):
            X_train, X_val = self.x_train[t_index], self.x_train[v_index]
            Y_train, Y_val = self.y_train[t_index], self.y_train[v_index]

            val_data=(X_val, Y_val)

            self.fit(X_train, Y_train, val_data)

            return self.model

    def test_model(self):
        from ml_statistics import BaseStatistics
        Y = np.zeros(self.y_test.shape)
        y_pred = self.model.predict(x=self.x_test, batch_size=8)
        stats = BaseStatistics(self.y_test, y_pred)
        return stats, y_pred

    def compile(self):
        self.loss_type = LOSS_TYPES[self.args.loss_type]
        self.optimizer_type = OPTIMIZER_TYPES[self.args.optimizer_type]
        self.model.compile(loss=self.loss_type, optimizer=self.optimizer_type)

    def fit(self, X, y, val_data):
        # print(X.shape)
        # print(y.shape)
        # print(self.args.batch_size)
        # print(self.args.epochs)
        # print(val_data)
        self.model.fit(x=X, y=y, batch_size=self.args.batch_size, epochs=self.args.epochs, validation_data=val_data, callbacks=self.calls, verbose=0)

    def define_calls(self):
        from keras import callbacks as C
        self.model_cnt+=1
        calls = list()
        calls.append( C.ModelCheckpoint(self.args.save_dir + '/weights-{}-'.format(self.model_cnt)+'{epoch:02d}.h5', save_best_only=True, save_weights_only=True, verbose=0) )
        calls.append( C.CSVLogger(self.args.save_dir + '/log.csv') )
        # calls.append( C.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs/{}'.format(actual_partition), batch_size=self.args.batch_size, histogram_freq=self.args.debug) )
        # calls.append( C.TensorBoard(log_dir=self.args.save_dir + '/tensorboard-logs/{}'.format(1), batch_size=self.args.batch_size, histogram_freq=self.args.debug) )
        calls.append( C.EarlyStopping(monitor='val_loss', patience=3, verbose=0) )
        calls.append( C.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001, verbose=0) )
        calls.append( C.LearningRateScheduler(schedule=lambda epoch: self.args.lr * (self.args.lr_decay ** ((1+epoch)/10) )) )
        # calls.append( C.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch)) )
    #    calls.append( C.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.)) )
        self.calls = calls
        return self.calls

    def get_input_data(self, mask=None):
        test_path = './db/{}-p{}-test.nucs'.format(self.args.organism, 1)
        train_path = './db/{}-p{}-train.nucs'.format(self.args.organism, 1)
        test_file = np.loadtxt(test_path, delimiter=';', dtype='int32')
        train_file = np.loadtxt(train_path, delimiter=';', dtype='int32')
        self.x_test, self.y_test = test_file[:, :-1], test_file[:, -1]
        self.x_train, self.y_train = train_file[:, :-1], train_file[:, -1]
        if mask is not None:
            self.x_train = self.x_train[:, mask.astype(bool)]
            self.x_test = self.x_test[:, mask.astype(bool)]

        self.input_size = self.x_test.shape[1]
        return self.x_test, self.y_test, self.x_train, self.y_train
