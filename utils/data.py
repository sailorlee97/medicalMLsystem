"""
@Time    : 2022/9/10 9:21
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: data.py
@Software: PyCharm
"""
import pandas as pd
import numpy as np
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader

class Data:
    def __init__(self,opt):
        self.opt = opt

    def getFormatIndustryFeatures(self,dataset):
        # trainNormal, test = obtainFeature(file_path)
        # dataset = pd.read_csv(file_path)
        # dataset = dataset.values
        dataset = self._preData(dataset)
        m, n = dataset.shape

        samples = dataset[:, 0:n - 1]
        labels = dataset[:, n - 1]
        x_tensor = torch.FloatTensor(samples)
        y_tensor = torch.FloatTensor(labels)
        y = torch.unsqueeze(y_tensor, 1)

        # samples = np.empty([batchsize, seq_length, num_feature])
        # labels =np.empty([batchsize, seq_length, 1])
        return [x_tensor, y]

    def _preData(self,x):
        scaler = MinMaxScaler()
        train_normal = scaler.fit_transform(x)

        return train_normal

    def finalObtainFeatures(self):
        df = pd.read_csv(self.opt.file)
        df_case = pd.read_csv(self.opt.filelabel)
        df['label'] = df_case['label']
        dd = df[df['label'] == 0]
        dd = dd.values

        # process malware

        nn = df[df['label'] == 1]
        # malware = malware.drop('Target')
        nn = nn.values
        # malware = np.delete(malware, -1, axis=1)
        #traindd = dd[:90,:]
        #testdd = dd[90:,:]

        #trainnn = nn[:75,:]
        #testnn = nn[:75, :]


        #train = np.vstack((testdd, testnn))

        #test = np.vstack((testdd, testnn))
        dd_ll = self.getFormatIndustryFeatures(dd)
        nn_ll = self.getFormatIndustryFeatures(nn)

        return dd_ll, nn_ll

    def make_dl(self,ll):
        ds = data.TensorDataset(ll[0], ll[1])
        train_loader = DataLoader(dataset=ds, batch_size=8,
                                  shuffle=True)
        return train_loader

    def finalgetdl(self):
        dd_ll,nn_ll = self.finalObtainFeatures()
        dd_dl = self.make_dl(dd_ll)
        nn_dl = self.make_dl(nn_ll)

        return dd_dl, nn_dl