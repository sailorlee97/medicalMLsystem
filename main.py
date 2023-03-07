"""
@Time    : 2022/9/10 15:22
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: main.py
@Software: PyCharm
"""
import os

from sklearn.metrics import classification_report,roc_auc_score

from modelTrainTest import trainModel
from options import Options
from superised import obtainNewData
from utils.data import Data
import pandas as pd
from tensorflow import keras
import numpy as np
from model.cnnmodel import cnnmodel
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from utils.plot_cm import plot_conf


def obtainnewfeatures(opt,dl,num):
    """
    Through VAE, data is enhanced and generated.

    :param opt:parameters
    :param dl:
    :param num: d
    """

    trainmodel = trainModel(opt)

    dfgeneratedd = pd.DataFrame()
    trainmodel.modelfit(dl)
    for i  in range(num):
        generatedd = trainmodel.gengerate(dl,'./savedmodel/vae_%s.pth'%opt.featurenamecsv)
        dfgeneratedd = dfgeneratedd.append(generatedd)

    dfgeneratedd.to_csv('./csv/%s.csv'%opt.featurenamecsv)

def runmodel(num_folds,inputs, targets,no_epochs):
    """

    :param num_folds:
    :param inputs:
    :param targets:
    :return:
    """
    acc_per_fold = []
    fold_no = 1
    inp_size = inputs.shape[1]
    cnn = cnnmodel()
    kfold = KFold(n_splits=num_folds, shuffle=True)
    for train, test in kfold.split(inputs, targets):

        model = cnn.build_model(inp_size)
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        history = model.fit(inputs[train], targets[train],
                            batch_size=64,
                            epochs=no_epochs,
                            verbose=1)
        scores = model.evaluate(inputs[test], targets[test], verbose=0)

        # scores = model.evaluate(x_test, y_t, verbose=1)

        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        y_pred = model.predict(inputs[test], batch_size=100)

        predict_label = np.argmax(y_pred, axis=1)

        # 或者
        y_t = np.argmax(targets[test], axis=-1)
        auc_score = roc_auc_score(y_t, predict_label)
        print('AUC:',auc_score)
        # y_t = keras.utils.to_categorical(predict_label, 2)
        # plot_conf(predict_label,y_t,['1p/19q-codeleted', '1p/19q-nocodeleted'])
        print(classification_report(y_t, predict_label, digits=4))
        acc_per_fold.append(scores[1] * 100)

        # Increase fold number
        fold_no = fold_no + 1

    return acc_per_fold

if __name__ == '__main__':
    import tensorflow as tf
    import keras.backend as KTF

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 限制GPU内存占用率
    sess = tf.compat.v1.Session(config=config)
    KTF.set_session(sess)  # 设置session

    opt = Options().parse()
    if os.path.exists('./csv/dd.csv') & os.path.exists('./csv/nn.csv'):

        x = obtainNewData('mutualfilter')
        label = x.pop('label')
        x_train = x.values
        # inp_size = x_train.shape[1]

        y_train = keras.utils.to_categorical(label, 2)

        x_train_labeled = np.expand_dims(x_train, axis=2)
        acc_per_fold = runmodel(5,x_train_labeled,y_train,20)

    else:
        newdata = Data(opt)
        dd_dl, nn_dl = newdata.finalgetdl()
        if opt.featurenamecsv == 'nn':
            obtainnewfeatures(opt, nn_dl, 9)
        else:
            obtainnewfeatures(opt, dd_dl, 9)

#  e = 0.988  sigma = 0.00221    [e-2*sigma,e+2*sigma]90%   [e-1*sigma,e+1*sigma]95%   0.988 [0.987,0.99]