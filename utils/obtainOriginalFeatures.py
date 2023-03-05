"""
@Time    : 2022/8/28 17:10
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: obtainOriginalFeatures.py
@Software: PyCharm
"""
import os
import pandas as pd
from radiomics import featureextractor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

class OriginalFeatures:
    def __init__(self,fileName,labelFile,isToCsv = True):
        self.fileName = fileName
        self.labelFile = labelFile
        self.df = pd.DataFrame()
        self.isToCsv = isToCsv

    def obtianFeatures(self):
        #g = os.walk(self.fileName)
        for (path, dir_path, file_list) in os.walk(self.fileName):
            imageName = ''
            maskName = ''
            for file_name in file_list:

                if file_name.endswith(".nii.gz"):
                    imageName = os.path.join(path, file_name)
                    # image = imageName.split("-")
                    # print(image[1])
                if file_name.endswith('.nii'):
                    maskName = os.path.join(path, file_name)
                    # mask = maskName.split("-")
                    # print(mask[1])
            print(imageName + "###" + maskName)
            if imageName != '' or maskName != '':
                extractor = featureextractor.RadiomicsFeatureExtractor()
                featureVector = extractor.execute('%s' % imageName, '%s' % maskName)
                df_add = pd.DataFrame.from_dict(featureVector.values()).T
                #print(df_add.values)
                df_add.columns = featureVector.keys()
                self.df = pd.concat([self.df, df_add])

        #if self.isToCsv:
        #    self.df.to_csv(self.fileName+'_feature_data_xz.csv')

        #return self.df

    def deleteExtaFeature(self):
        self.df.drop(index = [0],axis = 1)
        list = ['diagnostics_Versions_PyRadiomics','diagnostics_Versions_Numpy','diagnostics_Versions_SimpleITK',
                'diagnostics_Versions_PyWavelet','diagnostics_Versions_Python','diagnostics_Configuration_Settings',
                'diagnostics_Configuration_EnabledImageTypes','diagnostics_Image-original_Hash','diagnostics_Image-original_Dimensionality',
                'diagnostics_Image-original_Spacing','diagnostics_Image-original_Size','diagnostics_Mask-original_Hash',
                'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_Size','diagnostics_Mask-original_BoundingBox',
                'diagnostics_Mask-original_CenterOfMassIndex','diagnostics_Mask-original_CenterOfMass']
        for i in list:
            self.df =self.df.drop(i,axis = 1)

        if self.isToCsv:
            self.df.to_csv(self.fileName+'_feature_data_xz.csv')
        #return self.df

    def obtainLabel(self):
        df_case = pd.read_csv(self.labelFile)
        df_case_label = df_case['label']
        y = df_case_label.values
        return y

    def _preData(self,x):
        scaler = MinMaxScaler()
        train_normal = scaler.fit_transform(x)

        return train_normal

if __name__ == '__main__':
    utils = OriginalFeatures('../data/others','../data/TCIA_cases.csv',isToCsv=True)
    utils.obtianFeatures()

    utils.deleteExtaFeature()
  #  print(utils.df)
   # x = utils.featureSelect(utils.df)
    #print(x)