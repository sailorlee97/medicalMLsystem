# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:29:48 2018

@author: 13913
"""

from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D,Convolution1D
from keras import backend as K
from keras import optimizers
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score
from superised import gettestdata


x_train,y_train,x_test, y_test = gettestdata('mutualfilter')
inp_size =x_train.shape[1]
y_train = keras.utils.to_categorical(y_train, 2)
y_test  = keras.utils.to_categorical(y_test, 2)

x_train_labeled = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

model = Sequential()
model.add(Convolution1D(64,3,padding="same",activation="relu",input_shape=(inp_size,1)))
model.add(Convolution1D(64,3,padding="same",activation="relu"))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Convolution1D(128,3,padding="same",activation="relu"))
model.add(Convolution1D(128,3,padding="same",activation="relu"))
model.add(MaxPool1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))

model.add(Dense(2,activation="softmax"))
model.summary()
rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])
print(x_train_labeled.shape)

history = model.fit(x_train_labeled, y_train, epochs=50, batch_size=64,validation_split= 0.1)
scores = model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_test, batch_size=100)
auc_score = roc_auc_score(y_test,y_pred)
print(classification_report(y_test.argmax(-1), y_pred.argmax(-1), digits=4))

print("AUC:", auc_score)
print("CNN Accuracy: ", scores[1])