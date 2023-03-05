"""
@Time    : 2023/3/5 10:09
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: cnnmodel.py
@Software: PyCharm
"""
from tensorflow import keras
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
from superised import obtainNewData

class cnnmodel():
    def __init__(self):

        self.metrics = [
            # keras.metrics.TruePositives(name='tp'),
            # keras.metrics.FalsePositives(name='fp'),
            # keras.metrics.TrueNegatives(name='tn'),
            # keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            # keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]

    def  build_model(self,inp_size):
        model = Sequential()
        model.add(Convolution1D(64, 3, padding="same", activation="relu", input_shape=(inp_size, 1)))
        model.add(Convolution1D(64, 3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=(2)))
        model.add(Convolution1D(128, 3, padding="same", activation="relu"))
        model.add(Convolution1D(128, 3, padding="same", activation="relu"))
        model.add(MaxPool1D(pool_size=(2)))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))

        model.add(Dense(2, activation="softmax"))
        # model.summary()
        rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
        model.compile(loss='categorical_crossentropy',
                      optimizer=rmsprop,
                      metrics=['accuracy'])

        return model