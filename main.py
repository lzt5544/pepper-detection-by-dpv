import argparse
from multiprocessing import Process

from core import run

parser = argparse.ArgumentParser(description='')

parser.add_argument('--models', '-m', nargs='+', type=str, required=True, help='ann, xgb, svm, knn')
parser.add_argument('--config', '-c', type=str, required=True, help='the config path')
parser.add_argument('--train', '-t', action='store_true', help='execute training')
parser.add_argument('--validate', '-v', action='store_true', help='execute validation')

args = parser.parse_args()

config_path = args.config

run_train = args.train
run_validate = args.validate

for model in args.models:
    model = model.lower()
    if model not in ['xgb', 'knn', 'svm', 'ann']:
        raise KeyError('The mdoel can only accept [\'xgb\', \'knn\', \'svm\', \'ann\']')
    run(model, config_path, run_train, run_validate)