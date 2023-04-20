from tool import load_data, load_config
from core import fit_and_test_model, run


for method, config in load_config('config.json').items():
    print(f'run：{method}')
    hyperparameters = dict(config['models_choose'])
    extra_metrics = list(config['extra_metrics'])
    models_config = dict(config['models_config'])
    preserve_models = list(config['preserve_models'])
    data_path = config["data_path"]
    output_path = config['output_path']
    x, y = load_data(data_path)
    trainer = lambda  x, y : fit_and_test_model(models_config, x, y, hyperparameters, extra_metrics)
    excel = run(x, y, trainer, preserve_models)
    excel.to_excel(output_path)