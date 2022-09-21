"""
@Time    : 2022/9/10 15:22
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: main.py
@Software: PyCharm
"""
from modelTrainTest import trainModel
from options import Options
from utils.data import Data
import pandas as pd




def obtainnewfeatures(opt,dl,num):

    trainmodel = trainModel(opt)

    dfgeneratedd = pd.DataFrame()
    trainmodel.modelfit(dl)
    for i  in range(num):
        generatedd = trainmodel.gengerate(dl,'./savedmodel/vae_%s.pth'%opt.featurenamecsv)
        dfgeneratedd = dfgeneratedd.append(generatedd)

    dfgeneratedd.to_csv('./csv/%s.csv'%opt.featurenamecsv)

if __name__ == '__main__':
    opt = Options().parse()
    newdata = Data(opt)
    dd_dl, nn_dl = newdata.finalgetdl()
    if opt.featurenamecsv == 'nn':

        obtainnewfeatures(opt,nn_dl,9)

    else:
        obtainnewfeatures(opt, dd_dl, 9)