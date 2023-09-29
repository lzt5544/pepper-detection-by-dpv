import os
import pickle
import yaml

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor as tb

from models import SVM, XGB, KNN, ANN
from search_space import *
from tool import split_dataset, model_score


hyperparameters = {
    'xgb' : {XGB : xgb_space},
    'knn' : {KNN : knn_space},
    'svm' : {SVM : svm_space},
    'ann' : {ANN : ann_space}
}

def train(train_model, cfgs):
    predictor_cfgs = cfgs['predictor_cfgs']
    save_path = cfgs['save_path']
    train_data = cfgs['train_data']
    train_cfgs = cfgs['train_cfgs']

    predictor_cfgs['path'] = f'{save_path}/{train_model}'
    model = tb(**predictor_cfgs)
    model.fit(train_data,
            hyperparameters=hyperparameters[train_model], 
            fit_weighted_ensemble=False, 
            num_bag_sets=1,
            **train_cfgs)

def validate(train_model, cfgs):
    res = {}

    save_path = cfgs['save_path']
    num_bag_folds = cfgs['num_bag_folds']
    test_data = cfgs['test_data']
    metrics = cfgs['metrics']
    sort_by = cfgs['sort_by']
    sort_by = f'test_{sort_by}'
    problem_type = cfgs['predictor_cfgs']['problem_type']
    classes = None

    model_path = os.path.join(f'{save_path}/{train_model}', 'models', f'{train_model.upper()}_BAG_L1')
    
    if problem_type in ['binary', 'multiclass']:
        classes = set(cfgs['dataset'].iloc[:,-1])
        classes = list(classes)
        classes = sorted(classes)
        if isinstance(test_data.iloc[0, -1], str):
            test_data.iloc[:,-1] = np.array([classes.index(i) for i in test_data.iloc[:,-1].values])

    df_columns = [f'{s}_{metric}' for metric in metrics for s in ['train', 'val', 'test']]
    for name in os.listdir(model_path):
        if not '.' in name:
            print(name)
            scores_dir = dict([*zip(df_columns,[0]*len(df_columns))])
            for fold in range(1, num_bag_folds+1):
                score_each_fold = model_score(os.path.join(model_path, name, f'S1F{fold}'), test_data, metrics)
                for metric in metrics:
                    scores_dir[f'train_{metric}'] += score_each_fold[metric][0]
                    scores_dir[f'val_{metric}'] += score_each_fold[metric][1]
                    scores_dir[f'test_{metric}'] += score_each_fold[metric][2]
            for key, value in scores_dir.items():
                scores_dir[key] = value / num_bag_folds
            res.update({name : list(scores_dir.values())})

    res = pd.DataFrame(res).T
    
    res = res.set_axis(df_columns, axis=1)
    if sort_by is not None:
        res = res.sort_values(by=sort_by, ascending=False)
    res.to_excel(os.path.join(f'{save_path}/{train_model}', 'res.xlsx'))

def load_configs(config_path):
    cfgs = yaml.load(open(config_path, encoding='utf-8'), Loader=yaml.FullLoader)

    path = cfgs.get('data_path', None)
    predictor_cfgs = cfgs.get('predictor_cfgs', None)
    train_cfgs = cfgs.get('train_cfgs', None)
    test_size = cfgs.get('test_size', 0.2)
    metrics = cfgs.get('metrics', ['rmse'])
    train_cfgs = cfgs.get('train_cfgs', None)
    sort_by = cfgs.get('sort_by', None)

    dataset = pd.read_excel(path, index_col=0)
    train_data, test_data = split_dataset(dataset, test_size)

    save_path = predictor_cfgs['path']
    num_bag_folds = train_cfgs['num_bag_folds']

    return {
        'path' : path,
        'predictor_cfgs' : predictor_cfgs,
        'train_cfgs' : train_cfgs,
        'test_size' : test_size,
        'metrics' : metrics,
        'train_cfgs' : train_cfgs,
        'sort_by' : sort_by,
        'dataset' : dataset,
        'train_data' : train_data,
        'test_data' : test_data,
        'save_path' : save_path,
        'num_bag_folds' : num_bag_folds
    }

def run(train_model, config_path, run_train=True, run_validate=True):
    cfgs = load_configs(config_path)
    if run_train:
        train(train_model, cfgs)
    if run_validate:
        validate(train_model, cfgs)