import json
import pandas as pd


def load_config(path:str):
    with open(path, encoding='utf-8') as f:
        config = json.loads(f.read())
    return config

def load_data(path:str):
    data = pd.read_excel(path, index_col=0)
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    return x, y

def changed_columns_mapping(extra_metrics, prefix):
    res = {
        'score_test': f'{prefix}_loss'
    }
    for metric in extra_metrics:
        res.update({metric : f'{prefix}_{metric}'})
    return res