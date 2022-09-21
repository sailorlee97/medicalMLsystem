# coding=utf8
"""
@Time    : 2022/9/10 15:47
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: evalution.py
@Software: PyCharm
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score


def plt_loss(num_epochs,Loss_list):
    """
    :param num_epochs:训练轮数
    :param Loss_list: 损失函数的list
    :return:
    """
    x2 = range(0, num_epochs)
    # y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x2, y2, 'o-')
    # plt.title('Test accuracy vs. epoches')
    # plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('loss vs. epoches')
    plt.ylabel(' loss')
    plt.savefig("loss.png", dpi=1000)
    plt.show()

def evaluate(labels, scores,save = False, metric='roc'):
    if metric == 'roc':
        return roc(labels, scores,save = save)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels.cpu(), scores.cpu())
    else:
        raise NotImplementedError("Check the evaluation metric.")

def roc(labels, scores, saveto=None,save = False):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.

    fpr, tpr, _ = roc_curve(labels, scores)
    if save:
        np.savetxt("fpr.csv", fpr, delimiter=',')
        np.savetxt("tpr.csv", tpr, delimiter=',')
        #df_tpr = pd.DataFrame(tpr)
        #df = pd.concat(df_fpr,df_tpr)
        #df.to_csv('flowgananomaly.csv')
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels.cpu(), scores.cpu())
    return ap