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

from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from modelTrainTest import trainModel
from options import Options
from superised import obtainNewData, test_predictknn, test_predictDecisionTree, test_predictGaussianNB, \
    test_predictRandomForest, test_predictSVM
from utils.data import Data
import pandas as pd
from tensorflow import keras
import numpy as np
from model.cnnmodel import cnnmodel
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from utils.evalution import multi_models_roc
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

def run_othermodel():
    kfold = KFold(n_splits=5, shuffle=True)

    df = pd.read_csv('./data-addxz.csv')

    df_case = pd.read_csv('./TCIA_LGG_cases_159.csv')
    df['label'] = df_case['label']

    dataframe = df.replace([np.inf, -np.inf], np.nan).dropna()
    label = dataframe.pop('label')

    inputs = dataframe.values
    input_s = np.array(inputs)

    target_label = np.array(label)
    mean_tpr=[]
    mean_fpr=[]
    print('开始Knn算法运算')
    # Knn算法
    knn_tpr = 0.0  # 用来记录画平均ROC曲线的信息
    knn_fpr = np.linspace(0, 1, 100)

    for train, test in kfold.split(input_s, target_label):

        predictlabel = test_predictknn(input_s[train], target_label[train],input_s[test])
        # y_scores =  predictlabel[:, 1]

        fpr, tpr, thresholds = roc_curve(target_label[test], predictlabel, pos_label=1)
        knn_tpr += np.interp(knn_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        knn_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点

        auc_score = roc_auc_score(target_label[test], predictlabel)
        print('AUC:', auc_score)
    # #print(predicttrainlabel)
        print(classification_report(target_label[test],predictlabel))
    knn_tpr /= 5
    knn_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点

    print('开始DecisionTree算法运算')
    # DecisionTree算法
    DecisionTree_tpr = 0.0  # 用来记录画平均ROC曲线的信息
    DecisionTree_fpr = np.linspace(0, 1, 100)

    for train, test in kfold.split(input_s, target_label):

        predictlabel = test_predictDecisionTree(input_s[train], target_label[train],input_s[test])
        # y_scores =  predictlabel[:, 1]

        fpr, tpr, thresholds = roc_curve(target_label[test], predictlabel, pos_label=1)
        DecisionTree_tpr += np.interp(DecisionTree_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        DecisionTree_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点

        auc_score = roc_auc_score(target_label[test], predictlabel)
        print('AUC:', auc_score)
    # #print(predicttrainlabel)
        print(classification_report(target_label[test],predictlabel))
    DecisionTree_tpr /= 5
    DecisionTree_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点

    print('开始GaussianNB算法运算')
    # GaussianNB算法
    GaussianNB_tpr = 0.0  # 用来记录画平均ROC曲线的信息
    GaussianNB_fpr = np.linspace(0, 1, 100)

    for train, test in kfold.split(input_s, target_label):

        predictlabel = test_predictGaussianNB(input_s[train], target_label[train],input_s[test])
        # y_scores =  predictlabel[:, 1]

        fpr, tpr, thresholds = roc_curve(target_label[test], predictlabel, pos_label=1)
        GaussianNB_tpr += np.interp(GaussianNB_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        GaussianNB_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点

        auc_score = roc_auc_score(target_label[test], predictlabel)
        print('AUC:', auc_score)
    # #print(predicttrainlabel)
        print(classification_report(target_label[test],predictlabel))
    GaussianNB_tpr /= 5
    GaussianNB_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点

    print('开始RandomForest算法运算')
    # RandomForest算法
    RandomForest_tpr = 0.0  # 用来记录画平均ROC曲线的信息
    RandomForest_fpr = np.linspace(0, 1, 100)

    for train, test in kfold.split(input_s, target_label):
        predictlabel = test_predictRandomForest(input_s[train], target_label[train], input_s[test])
        # y_scores =  predictlabel[:, 1]

        fpr, tpr, thresholds = roc_curve(target_label[test], predictlabel, pos_label=1)
        RandomForest_tpr += np.interp(RandomForest_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        RandomForest_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点

        auc_score = roc_auc_score(target_label[test], predictlabel)
        print('AUC:', auc_score)
        # #print(predicttrainlabel)
        print(classification_report(target_label[test], predictlabel))
    RandomForest_tpr /= 5
    RandomForest_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点

    mean_fpr.append(knn_fpr)
    mean_fpr.append(DecisionTree_fpr)
    mean_fpr.append(GaussianNB_fpr)
    mean_fpr.append(RandomForest_fpr)
    mean_tpr.append(knn_tpr)
    mean_tpr.append(DecisionTree_tpr)
    mean_tpr.append(GaussianNB_tpr)
    mean_tpr.append(RandomForest_tpr)
    return mean_fpr,mean_tpr

def run_cnnmodel(num_folds,inputs, targets,no_epochs):
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
    mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
    mean_fpr = np.linspace(0, 1, 100)
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
        y_scores = y_pred[:, 1]

        predict_label = np.argmax(y_pred, axis=1)
        fpr, tpr, thresholds = roc_curve(targets[test].argmax(-1), y_scores, pos_label=1)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        mean_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点

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
    mean_tpr /= num_folds
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点

    return acc_per_fold,mean_fpr,mean_tpr

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
        # acc_per_fold = runmodel(5,x_train_labeled,y_train,20)

        acc_per_fold,cnn_fpr,cnn_tpr = run_cnnmodel(5,x_train_labeled,y_train,20)
        other_fpr,other_tpr=run_othermodel()
        # trainValue, testValue, trainlabels, testlabels = train_test_split(x_train, y_train, test_size=0.25,
        #                                                                   random_state=0)
        # predictlabel = test_predictknn(trainValue, trainlabels, testValue)
        # predictlabel = test_predictDecisionTree(trainValue, trainlabels, testValue)
        # print(testlabels.argmax(-1))
        # print("----------------------------------")
        # print(predictlabel.argmax(-1))
        # knn_fpr, knn_tpr, thresholds = roc_curve(testlabels.argmax(-1), predictlabel.argmax(-1), pos_label=1)


        names = ['Cnn','Knn','DecisionTree','GaussianNB','RandomForest']
        fpr_total = []
        tpr_total = []
        fpr_total.append(cnn_fpr)
        fpr_total.extend(other_fpr)
        tpr_total.append(cnn_tpr)
        tpr_total.extend(other_tpr)
        # color:'crimson','orange','gold','mediumseagreen','steelblue', 'mediumpurple'
        colors = ['crimson','orange','gold','mediumseagreen','steelblue']

        # ROC curves
        train_roc_graph = multi_models_roc(names, colors, fpr_all=fpr_total, tpr_all=tpr_total, save=True)
        # train_roc_graph.savefig('ROC_Train_all.png')
        print("end")



    else:
        newdata = Data(opt)
        dd_dl, nn_dl = newdata.finalgetdl()
        if opt.featurenamecsv == 'nn':
            obtainnewfeatures(opt, nn_dl, 9)
        else:
            obtainnewfeatures(opt, dd_dl, 9)

#  e = 0.988  sigma = 0.00221    [e-2*sigma,e+2*sigma]90%   [e-1*sigma,e+1*sigma]95%   0.988 [0.987,0.99]