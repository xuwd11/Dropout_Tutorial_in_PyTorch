import sys

import warnings
warnings.filterwarnings("ignore")

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.regularizers import l2

from keras.utils import to_categorical

from keras.models import load_model

import numpy as np
import pandas as pd
import time
import h5py

import pickle

class IO:
    def __init__(self, file_name):
        self.file_name = file_name
        
    def to_pickle(self, obj):
        with open(self.file_name, 'wb') as output:
            pickle.dump(obj, output, protocol=pickle.HIGHEST_PROTOCOL)
    
    def read_pickle(self):
        with open(self.file_name, 'rb') as input_:
            obj = pickle.load(input_)
        return obj
    
f = h5py.File('data/train.h5', 'r')
data_train = np.concatenate((f['data'].value, f['label'].value), axis=1)
f.close()
X_train = data_train[:, 0].reshape(-1, 1)
y_train = data_train[:, 1].reshape(-1, 1)
print(data_train.shape)

f = h5py.File('data/test.h5', 'r')
data_test = np.concatenate((f['data'].value, f['label'].value), axis=1)
f.close()

X_test = np.arange(-1.72, 3.51, 0.01).reshape(-1, 1)

class FFNN2:
    def __init__(self, hidden_layers=[1024, 1024, 1024, 1024], droprate=0.1, activation='relu'):
        reg = 1e-6
        model = Sequential()
        model.add(Dense(hidden_layers[0], activation=activation, input_shape=(1, ), kernel_initializer='lecun_uniform', \
                        W_regularizer=l2(reg)))
        for d in hidden_layers[1:]:
            model.add(Dropout(droprate))
            model.add(Dense(d, activation=activation, kernel_initializer='lecun_uniform', \
                            W_regularizer=l2(reg)))
        model.add(Dropout(droprate))
        model.add(Dense(1, W_regularizer=l2(reg)))
        self.model = model
        
    def fit(self, X_train, y_train, lr=0.0001, epochs=1000000, batch_size=100, verbose=0):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))
        self.result = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self
    
def predict(model, X, T=10000):
    # Reference: https://github.com/yaringal/DropoutUncertaintyExps/blob/master/bostonHousing/net/net/net.py
    standard_pred = model.predict(X)
    predict_stochastic = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    Yt_hat = np.array([predict_stochastic([X, 1]) for _ in range(T)]).squeeze()
    return standard_pred, Yt_hat

start = time.time()
ffnn2_relu = FFNN2(activation='relu').fit(X_train, y_train, verbose=1)
ffnn2_relu.model.save('results/co2_regression_MC_relu.h5')
print('Time used: {}'.format(time.time()-start))

start = time.time()
model = load_model('results/co2_regression_MC_relu.h5')
y_pred, y_hat = predict(model, X_test)
y_mc = y_hat.mean(axis=0)
y_mc_std = y_hat.std(axis=0)
IO('results/co2_regression_MC_relu_results.pkl').to_pickle((y_pred, y_mc, y_mc_std))
print('Time used: {}'.format(time.time()-start))