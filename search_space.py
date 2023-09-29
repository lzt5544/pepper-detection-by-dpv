from autogluon.common import space

svm_space = {
    'C' : space.Real(lower=3e-3, upper=3e3, default=0.1, log=True),
    'kernel' : space.Categorical('linear', 'rbf', 'poly', 'sigmoid'),
    # 'epsilon' : space.Real(lower=1e-3, upper=1e3, default=0.01, log=True),
    # 'gamma' : space.Real(lower=1e-2, upper=1e2, default=0.01, log=True),
    # 'degree' : space.Categorical(2, 3, 4),
    # 'coef0' : space.Categorical(-1, 0, 1)
}

ann_space = {
    'num_epochs' : 500,
    'epochs_wo_improve': 20,
    'activation': 'relu',
    'embedding_size_factor': 1.0,
    'embed_exponent': 0.56,
    'max_embedding_dim': 100,
    'optimizer': 'adam',
    'proc.embed_min_categories': 4,
    'proc.impute_strategy': 'median',
    'proc.max_category_levels': 100,
    'proc.skew_threshold': 0.99,
    'use_ngram_features': False,
    'max_batch_size': 512,
    'loss_function': 'auto',
    'use_batchnorm': space.Categorical(False, True),
    'learning_rate' : space.Real(1e-4, 3e-2, default=3e-4, log=True),
    'dropout_prob': space.Categorical(0.1, 0.0, 0.5, 0.2, 0.3, 0.4),
    'num_layers' : space.Categorical(2, 3, 4),
    'hidden_size' : space.Categorical(128, 256, 512),
    "weight_decay": space.Real(1e-12, 1.0, default=1e-6, log=True),
    "gamma": space.Real(0.1, 10.0, default=5.0),
    "alpha": space.Categorical(0.001, 0.01, 0.1, 1.0),
}

knn_space = {
    'n_neighbors' : space.Int(lower=3, upper=10),
    'weights' : space.Categorical('uniform', 'distance'),
    'algorithm' : space.Categorical('auto', 'ball_tree', 'kd_tree', 'brute'),
    'p' : space.Categorical(1, 2),
    'leaf_size' : space.Int(lower=30, upper=100)
}

xgb_space = {
    'n_estimators' : space.Int(lower=10, upper=1000, default=500),
    'learning_rate' : space.Real(lower=5e-3, upper=0.2, default=0.1, log=True),
    'max_depth' : space.Int(lower=3, upper=10, default=6),
    'min_child_weight' : space.Int(lower=1, upper=100, default=1),
    'reg_lambda' : space.Real(lower=1e-3, upper=1e2, default=0.1, log=True),
    'reg_alpha' : space.Real(lower=1e-3, upper=1e2, default=0.1, log=True),
    'gamma' : space.Int(lower=0, upper=10, default=5)
}