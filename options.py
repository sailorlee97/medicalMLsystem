"""
@Time    : 2022/6/23 14:47
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: options.py
@Software: PyCharm
"""
import argparse
import os


class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--csv', default=True, help='exist csv--true,else -- flase')
        self.parser.add_argument('--file',default='./data/data-addxz.csv',help='yes - model train;no - we will load model trained.')
        self.parser.add_argument('--filelabel',default='./data/TCIA_LGG_cases_159.csv',help='data class labels.')
        self.parser.add_argument('--num_epochs', default=1500, help='data class labels.')
        self.parser.add_argument('--load_weights', default=False, help='data class labels.')
        self.parser.add_argument('--batchsize', default=8, help='data class labels.')
        self.parser.add_argument('--device', default='cuda', help='data class labels.')
        self.parser.add_argument('--metric', default='roc', help='data class labels.')
        self.parser.add_argument('--featurenamecsv', default='dd', help='data class labels.')

        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        return self.opt