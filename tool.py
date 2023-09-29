import copy
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, log_loss


def root_mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    return mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput, squared=False)

_precision_score = precision_score
def precision_score(y_true, y_pred, *, labels=None, pos_label=1, sample_weight=None, zero_division="warn"):
    if len(set(y_true)) > 2:
        return _precision_score(y_true, y_pred, labels=labels, pos_label=pos_label, average='micro', sample_weight=sample_weight, zero_division=zero_division)
    else:
        return _precision_score(y_true, y_pred, labels=labels, pos_label=pos_label, average='binary', sample_weight=sample_weight, zero_division=zero_division)

RANDOM_STATE = 42

metrics_dir = {
    'r2' : r2_score,
    'rmse' : root_mean_squared_error,
    'mse' : mean_squared_error,
    'acc' : accuracy_score,
    'pre' : precision_score,
    'rec' : recall_score,
    'f1' : f1_score,
    'auc' : roc_auc_score,
    'log_loss' : log_loss
}

def split_dataset(df, test_size):
    np.random.seed(RANDOM_STATE)
    shuffled_indices = np.random.permutation(len(df))
    split_index = int((1-test_size) * len(df))

    train_indices = shuffled_indices[:split_index]
    test_indices = shuffled_indices[split_index:]

    train_data = df.iloc[train_indices, :]
    test_data = df.iloc[test_indices, :]

    return train_data, test_data

def model_score(path, data_test, metrics):
    with open(os.path.join(path, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    data_train = pd.read_pickle(os.path.join(path, 'data_train.pkl'))
    data_val = pd.read_pickle(os.path.join(path, 'data_val.pkl'))

    res = {}

    for metric in metrics:
        score = metrics_dir[metric]
        if metric == 'log_loss': 
            y_pred_train = model.predict_proba(data_train.iloc[:, :-1])
            y_pred_val = model.predict_proba(data_val.iloc[:, :-1])
            y_pred_test = model.predict_proba(data_test.iloc[:, :-1])
        else:
            y_pred_train = model.predict(data_train.iloc[:, :-1])
            y_pred_val = model.predict(data_val.iloc[:, :-1])
            y_pred_test = model.predict(data_test.iloc[:, :-1])
        # print(data_test.iloc[:,-1].shape, y_pred_test.shape)
        train_score = score(data_train.iloc[:,-1], y_pred_train)
        val_score =  score(data_val.iloc[:,-1], y_pred_val)
        test_score =  score(data_test.iloc[:,-1], y_pred_test)

        res.update({metric : [train_score, val_score, test_score]})
    res.update({'hyperparameters' : model.get_params()['hyperparameters']})

    return res

def find_best_para(path):
    res = None
    for file_name in os.listdir(path):
        excel_file_path = os.path.join(path, file_name, 'res.xlsx')
        if os.path.exists(excel_file_path):
            model_performance = pd.read_excel(excel_file_path, index_col=0)
            best = model_performance.iloc[0, :]
            best_model_path = os.path.join(path, file_name, f'models/{file_name.upper()}_BAG_L1/{model_performance.index[0]}/S1F1/model.pkl')
            with open(best_model_path, 'br') as f:
                model = pickle.load(f)
            best_para = model.get_info()['hyperparameters']
            res_dir = dict([*zip(model_performance.columns, best)])
            res_dir.update({'best_para' : str(best_para)})
            # print(pd.DataFrame(res_dir))
            res = pd.DataFrame(res_dir, index=[file_name]) if res is None else pd.concat((res, pd.DataFrame(res_dir, index=[file_name])))
    res.to_excel(os.path.join(path, 'best_models_performance.xlsx'))
