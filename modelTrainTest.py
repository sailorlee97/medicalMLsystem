# coding=utf8
"""
@Time    : 2022/8/28 21:37
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: modelTrainTest.py
@Software: PyCharm
"""
import os
import pandas as pd
import torch
import numpy as np
from utils.obtainOriginalFeatures import OriginalFeatures
from sklearn.preprocessing import MinMaxScaler
from options import Options
from model.VAE import VAE
from utils.data import Data
from utils.evalution import plt_loss,evaluate

class trainModel:

    def __init__(self,opt):

        self.opt = opt
        self.device = 'cuda'
        self.input = torch.empty(size=(self.opt.batchsize, 112), dtype=torch.float32,
                                 device=self.device)
        self.gt = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        self.fixed_input = torch.empty(size=(self.opt.batchsize, 112), dtype=torch.float32,
                                   device=self.device)
        self.total_steps = 0

    def calculate_losses(self,x, preds):
        losses = np.zeros(len(x))
        for i in range(len(x)):
            losses[i] = ((preds[i] - x[i]) ** 2).mean(axis=None)

        return losses

    def set_input(self, x,y):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(x.size()).copy_(x[0])
            self.gt.resize_(y.size())

            # Copy the first batch as the fixed input.

#    def selectFeature(self):
#        if os.access("./data/-modsegment_feature_data_xz.csv", os.F_OK):
#            df = pd.read_csv('./data/-modsegment_feature_data_xz.csv',encoding='gb18030')
#            df.drop(0, axis=1)  # 删除第0列
#            scaler = MinMaxScaler()
#            train_normal = scaler.fit_transform(df)

#            return train_normal
#        else:
#            utils = OriginalFeatures('./data/-modsegment','./data/TCIA_LGG_cases_159.csv',isToCsv=True)
#            utils.deleteExtaFeature()
#            scaler = MinMaxScaler()
#            train_normal = scaler.fit_transform(utils.df)

#            return train_normal

    def modelfit(self,train_dl):
        #dataset = Data(self.opt)
        #dd_dl, nn_dl = dataset.finalgetdl()

        model = VAE(112, 112).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), 0.0001)
        loss_func = torch.nn.MSELoss(reduction='mean')

        Loss_list = []

        for epoch in range(self.opt.num_epochs):
            total_loss = 0.
            for step, (x,y) in enumerate(train_dl):

                x = x.to(self.device)
                x_recon, z, mu, logvar = model.forward(x)

                # loss = calculate_losses(x_recon,x)
                loss = loss_func(x_recon, x)
                optimizer.zero_grad()
                # 计算中间的叶子节点，计算图
                loss.backward()
                # 内容信息反馈
                optimizer.step()
                total_loss += loss.item() * len(x)
                # print('Epoch :', epoch, ';Batch', step, ';train_loss:%.4f' % loss.data)
                # writer.add_scalar('loss', loss, step + len(train_loader) * epoch)  # 可视化变量loss的值
            #total_loss /= len(X_normal)
            print('Epoch {}/{} : loss: {:.4f}'.format(
                epoch + 1, self.opt.num_epochs, loss.item()))
            Loss_list.append(loss.item())

        plt_loss(self.opt.num_epochs,Loss_list)

        torch.save(model.state_dict(), './savedmodel/vae_%s.pth'%self.opt.featurenamecsv)

        return model

    def gengerate(self,samples,PATH):

        model = VAE(112, 112).to(self.device)
        model.load_state_dict(torch.load(PATH))

        df = pd.DataFrame()
        with torch.no_grad():


            for i, (x,y) in enumerate(samples, 0):

                x =x.to(self.device)
                testing_set_predictions, z, mu, logvar = model.forward(x)
                dfnew = pd.DataFrame(testing_set_predictions.cpu().numpy())
                df = df.append(dfnew)

                # if i == 0:
                #     a = testing_set_predictions.cpu().numpy()
                # else:
                #     a.e
            print(df.values.shape)
        return df

    def predict(self,clf):
        dataset = Data(self.opt)
        train_dl, test_dl = dataset.finalgetdl()

        if self.opt.load_weights:
            path_g = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
            pretrained_dict_g = torch.load(path_g)['state_dict']

            path_d = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
            pretrained_dict_d = torch.load(path_d)['state_dict']
            try:
                self.netg.load_state_dict(pretrained_dict_g)
                self.netd.load_state_dict(pretrained_dict_d)
            except IOError:
                raise IOError("netG or netD weights not found")
            print('   Loaded weights.')

        self.an_scores = torch.zeros(size=(len(test_dl.dataset),), dtype=torch.float32, device=self.device)
        self.gt_labels = torch.zeros(size=(len(test_dl.dataset),1), dtype=torch.long, device=self.device)

        with torch.no_grad():
            # self.times = []

            for i, (x,y) in enumerate(test_dl, 0):

                x =x.to(self.device)
                testing_set_predictions, z, mu, logvar = clf.forward(x)
                # print(testing_set_predictions)
                #testing_set_predictions = testing_set_predictions.detach().cpu().numpy()
                error = torch.mean(torch.pow((testing_set_predictions - x), 2), dim=1)

                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = y.reshape(y.size())

            #print(self.an_scores)

            scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            print('scores:',scores)
            print('label:',self.gt_labels)
            auc = evaluate(self.gt_labels,scores, metric=self.opt.metric, save=False)
            print(auc)
        return auc
