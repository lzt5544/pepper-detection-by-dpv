import os

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from autogluon.tabular import TabularPredictor as tb

from tool import *
from preprocessing import *


def get_res(res, useful_columns, prefix):
    extra_metrics = useful_columns[2:]
    return res.loc[:,useful_columns].rename(columns=changed_columns_mapping(extra_metrics, prefix)).set_index(res.iloc[:,0].values).drop(columns=['model'])

def split_dataset(x, y, test_size=0.2, random_state=0, fold_num=3):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    data_test = pd.concat((x_test,y_test),axis=1)
    skf = KFold(n_splits=fold_num, shuffle=True, random_state=random_state)
    dataset = []
    for train_idx, val_idx in skf.split(x_train, y_train):
        train_data = pd.concat((x_train.iloc[train_idx],y_train.iloc[train_idx]),axis=1)
        val_data = pd.concat((x_train.iloc[val_idx],y_train.iloc[val_idx]),axis=1)
        dataset.append([train_data,val_data])
    return dataset, data_test

def fit_and_test_model(models_config, x, y, hyperparameters, extra_metrics):
    dataset, data_test = split_dataset(x, y)
    r = []
    for data_train, data_val in dataset:
        models = tb(**models_config)
        models.fit(data_train,hyperparameters=hyperparameters)
        train_res = models.leaderboard(data_train,silent=True,extra_metrics=extra_metrics)
        cv_res = models.leaderboard(data_val,silent=True,extra_metrics=extra_metrics)
        test_res = models.leaderboard(data_test,silent=True,extra_metrics=extra_metrics)
        useful_columns = ['model','score_test']
        useful_columns.extend(extra_metrics)
        train_res = get_res(train_res, useful_columns, 'train')
        cv_res = get_res(cv_res, useful_columns, 'cv')
        test_res = get_res(test_res, useful_columns, 'test')
        r.append(pd.concat((train_res,cv_res,test_res), axis=1))
    return (r[0] + r[1] + r[2]) / 3

def run(x, y, trainer, model_name):
    res = {}
    print('raw')
    res.update({'raw':trainer(x, y)})
    for method in [SG,WT,MMS,SS,SNV,D1,D2]:
        print(method.__name__)
        x_temp = pd.DataFrame(method(x.values))
        res.update({method.__name__:trainer(x_temp, y)})
    for method1 in [SG,WT]:
        for method2 in [MMS,SS,SNV,D1,D2]:
            print(method1.__name__,'-',method2.__name__)
            x_temp = pd.DataFrame(method2(method1(x.values)))
            res.update({f'{method1.__name__}-{method2.__name__}':trainer(x_temp, y)})
    for method1 in [SG,WT]:
        for method2 in [MMS,SS]:
            for method3 in [SNV,D1,D2]:
                print(method1.__name__,'-',method2.__name__,'-',method3.__name__)
                x_temp = pd.DataFrame(method3(method2(method1(x.values))))
                res.update({f'{method1.__name__}-{method2.__name__}-{method3.__name__}':trainer(x_temp, y)})
    excel = None
    for key,she in res.items():
        temp = pd.concat((she.loc[model_name,:], pd.DataFrame([key]*len(model_name), index=model_name, columns=['method'])), axis=1)
        excel = temp if excel is None else pd.concat((excel,temp))
    return excel