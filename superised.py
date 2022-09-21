"""
@Time    : 2022/9/11 13:59
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: superised.py
@Software: PyCharm
"""
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
# import seaborn as sns
# import matplotlib.pyplot as plt
import os

# plt.style.use('seaborn-colorblind')
# %matplotlib inline
from feature_selection import filter_method as ft
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

def valtrail():
    df = pd.read_csv('./data-addxz.csv')

    df_case = pd.read_csv('./TCIA_LGG_cases_159.csv')

    corr = ft.corr_feature_detect(data=df,threshold=0.9)
    # print all the correlated feature groups!
    for i in corr:
        print(i,'\n')

def mutualfilter():

    df = pd.read_csv('./data-addxz.csv')
    df_case = pd.read_csv('./TCIA_LGG_cases_159.csv')
    dflabel = df_case['label']
    mi = ft.mutual_info(X=df, y=dflabel, select_k=0.2)
    mi.tolist()

    return mi

def _obtainfeatures(lists,data):
    """

    :param lists
    :param data
    :return:
    """
    for i in lists:
        datadf = data[lists]

    return data

def getseries(series):
    """

    :param series:
    :return:
    """
    list = []
    for x,y in series.iteritems():
        list.append(x)
    return list

def Univariatefiler():

    df = pd.read_csv('./data-addxz.csv')
    df_case = pd.read_csv('./TCIA_LGG_cases_159.csv')

    dftrain = df.iloc[:155,:]
    dftrainlabel = df_case.iloc[:155,:]
    dftrainlabel = dftrainlabel['label']

    dftest = df.iloc[155:,:]
    dftestlabel = df_case.iloc[155:,:]
    dftestlabel = dftestlabel['label']

    uni_roc_auc = ft.univariate_roc_auc(X_train=dftrain, y_train=dftrainlabel,
                                        X_test=dftest, y_test=dftestlabel, threshold=0.6)


    return uni_roc_auc

def Forwardwrapper():
    df = pd.read_csv('./data-addxz.csv')
    df_case = pd.read_csv('./TCIA_LGG_cases_159.csv')

    dftrain = df.iloc[:155, :]
    dftrainlabel = df_case.iloc[:155, :]
    dftrainlabel = dftrainlabel['label']

    dftest = df.iloc[155:, :]
    dftestlabel = df_case.iloc[155:, :]
    dftestlabel = dftestlabel['label']
    sfs1 = SFS(RandomForestClassifier(n_jobs=-1, n_estimators=5),
               k_features=10,
               forward=True,
               floating=False,
               verbose=1,
               scoring='roc_auc',
               cv=3)

    sfs1 = sfs1.fit(np.array(dftrain), dftrainlabel)
    selected_feat1 = dftrain.columns[list(sfs1.k_feature_idx_)]
    print(selected_feat1)

    return selected_feat1

def BackwardElimination():

    df = pd.read_csv('./data-addxz.csv')
    df_case = pd.read_csv('./TCIA_LGG_cases_159.csv')

    dftrain = df.iloc[:155, :]
    dftrainlabel = df_case.iloc[:155, :]
    dftrainlabel = dftrainlabel['label']

    sfs1 = SFS(RandomForestClassifier(n_jobs=-1, n_estimators=5),
               k_features=10,
               forward=True,
               floating=False,
               verbose=1,
               scoring='roc_auc',
               cv=3)

    sfs2 = sfs1.fit(np.array(dftrain.fillna(0)), dftrainlabel)
    selected_feat2 = dftrain.columns[list(sfs2.k_feature_idx_)]
    print(selected_feat2)

    return selected_feat2

def ExhaustiveFeaturewrapper():

    df = pd.read_csv('./data-addxz.csv')
    df_case = pd.read_csv('./TCIA_LGG_cases_159.csv')

    dftrain = df.iloc[:155, :]
    dftrainlabel = df_case.iloc[:155, :]
    dftrainlabel = dftrainlabel['label']

    efs1 = EFS(RandomForestClassifier(n_jobs=-1, n_estimators=5, random_state=0),
               min_features=1,
               max_features=6,
               scoring='roc_auc',
               print_progress=True,
               cv=2)
    efs1 = efs1.fit(np.array(dftrain[dftrain.columns[0:10]].fillna(0)), dftrainlabel)
    selected_feat3 = dftrain.columns[list(efs1.best_idx_)]
    print(selected_feat3)
    return selected_feat3

def gettestdata(featureselection):
    scaler = MinMaxScaler()
    df = pd.read_csv('./data-addxz.csv')
    df_case = pd.read_csv('./TCIA_LGG_cases_159.csv')

    if featureselection == 'Univariate':
        featuresseries = Univariatefiler()
        featureslist = getseries(featuresseries)
        df = df[featureslist]
    elif featureselection == 'mutualfilter':
        featureslist = mutualfilter()
        featureslist.tolist()
        df = df[featureslist]
    elif featureselection == 'Forwardwrapper':
        featureslist = Forwardwrapper()
        featureslist.tolist()
        df = df[featureslist]
    elif featureselection == 'BackwardElimination':
        featureslist = BackwardElimination()
        featureslist.tolist()
        df = df[featureslist]
    elif featureselection == 'ExhaustiveFeaturewrapper':
        featureslist = ExhaustiveFeaturewrapper()
        featureslist.tolist()
        df = df[featureslist]
    else:
        print('features stay!')

    print(df)

    df['label'] = df_case['label']
    dd = df[df['label'] == 0]
    dd = dd.values

    m_dd,n_dd = dd.shape
    ddValue = dd[:, 0:n_dd - 1]
    ddValue = scaler.fit_transform(ddValue)

    nn = df[df['label'] == 1]
    # malware = malware.drop('Target')
    nn = nn.values
    m_nn,n_nn = nn.shape
    nnValue = nn[:, 0:n_nn - 1]
    nnValue = scaler.fit_transform(nnValue)

    #nn = scaler.fit_transform(nn)
    # malware = np.delete(malware, -1, axis=1)
    traindd = ddValue[:98,:]
    testdd = ddValue[98:,:]

    df_casenn = df_case[df_case['label'] == 1]
    df_casenn = df_casenn.values
    df_casedd = df_case[df_case['label'] == 0]
    df_casedd = df_casedd.values

    trainddlabel = df_casedd[:98]
    testddlabel= df_casedd[98:]

    trainnn = nnValue[:80,:]
    testnn = nnValue[80:, :]

    trainnnlabel = df_casenn[:80]
    testnnlabel= df_casenn[80:]

    train = np.vstack((traindd, trainnn))
    #np.random.shuffle(train)
    test = np.vstack((testdd, testnn))

    trainlabels = np.vstack((trainddlabel, trainnnlabel))
    xtrain,ytrain = trainlabels.shape
    trainlabels = trainlabels[:,ytrain-1]

    testlabels = np.vstack((testddlabel, testnnlabel))
    xtest, ytest = testlabels.shape
    testlabels = testlabels[:, ytest - 1]
    #np.random.shuffle(test)

    # m, n = train.shape
    # trainValue = train[:, 0:n - 1]
    # trainValue = scaler.fit_transform(trainValue)
    # labels = train[:, n - 1]
    #
    # m_test, n_test = test.shape
    # testValue = test[:, 0:n_test - 1]
    # testValue = scaler.fit_transform(testValue)
    # testlabels = test[:, n_test- 1]
    trainlabels = trainlabels.astype('int')
    testlabels = testlabels.astype('int')

    Value, y = obtainNewData(featureselection)
    train = np.vstack((train,Value))
    #newlabel = np.append(ydd,ynn)
    #ydd = ydd.astype('int')
    #ynn = ynn.astype('int')
    trainlabels = np.append(trainlabels,y)

    return train,trainlabels,test,testlabels

def _selectfeature(featureselection,df):

    if featureselection == 'Univariate':
        featuresseries = Univariatefiler()
        featureslist = getseries(featuresseries)
        df = df[featureslist]
    elif featureselection == 'mutualfilter':
        featureslist = mutualfilter()
        featureslist.tolist()
        df = df[featureslist]
    elif featureselection == 'Forwardwrapper':
        featureslist = Forwardwrapper()
        featureslist.tolist()
        df = df[featureslist]
    elif featureselection == 'BackwardElimination':
        featureslist = BackwardElimination()
        featureslist.tolist()
        df = df[featureslist]
    elif featureselection == 'ExhaustiveFeaturewrapper':
        featureslist = ExhaustiveFeaturewrapper()
        featureslist.tolist()
        df = df[featureslist]
    else:
        df =df
        print('features stay!')

    return df

def obtainNewData(featureselection):

    dd = pd.read_csv('./csv/dd.csv')
    dd = _selectfeature(featureselection,dd)
    ddValue = dd.values
    #m_ddValue, n_ddvalue = ddValue.shape
    #ddValue= ddValue[:, 0: n_ddvalue - 1]
    ydd = np.zeros(len(ddValue))

    nn = pd.read_csv('./csv/nn.csv')
    nn = _selectfeature(featureselection, nn)
    nnValue = nn.values
    #m_nnValue, n_nnvalue = nnValue.shape
    #nnValue= nnValue[:, 0: n_nnvalue - 1]
    ynn = np.ones(len(nnValue))

    newtrain = np.vstack((ddValue, nnValue))
    newlabel = np.append(ydd, ynn)

    return newtrain,newlabel

def test_predictknn(trainValue,labels,testValue):
    neigh =KNeighborsClassifier(n_neighbors=3)

    neigh.fit(trainValue,labels)
    predicttrainlabel = neigh.predict(trainValue)
    label = neigh.predict(testValue)
    return label,predicttrainlabel

def test_predictDecisionTree(trainValue,labels,testValue):

    clf = DecisionTreeClassifier()
    clf.fit(trainValue, labels)
    label = clf.predict(testValue)
    return label

def test_predictGaussianNB(trainValue,labels,testValue):

    clf = GaussianNB(priors=None)
    clf.fit(trainValue, labels)
    label = clf.predict(testValue)
    return label

def test_predictRandomForest(trains,labels,tests):

    clf = RandomForestClassifier(random_state=0)
    clf.fit(trains, labels)
    #train
    predicttrainlabel = clf.predict(trains)
    label = clf.predict(tests)
    return label,predicttrainlabel

def test_predictSVM(trainValue,labels,testValue):

    clf = svm.SVC(gamma=0.001)
    clf.fit(trainValue, labels)
    label = clf.predict(testValue)
    return label

if __name__ == '__main__':
    # a = Forwardwrapper()
    # b = BackwardElimination()
    #c = ExhaustiveFeaturewrapper()
    #a = mutualfilter()
    trainValue,labels,testValue,testlabels = gettestdata('Univariate')
    predictlabel = test_predictSVM(trainValue,labels,testValue)
    #print(predicttrainlabel)
    print(classification_report(testlabels,predictlabel))
    #ddValue,ydd,nnValue,ynn = obtainNewData()
    #ytrain = np.append(ydd,ynn)
    #print(ytrain)
