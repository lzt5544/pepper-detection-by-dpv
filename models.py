import os
import threading
import time
import math
import logging

import numpy as np
import pandas as pd
from autogluon.common import space
from autogluon.core.models import AbstractModel
from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel


logger = logging.getLogger(__name__)

def save_split_data(func):
    def wrapper(self, X, y, X_val=None, y_val=None, **kwargs):

        func(self, X, y, X_val=None, y_val=None, **kwargs)
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        def save():
            pd.concat([X, y], axis=1).to_pickle(os.path.join(self.path, 'data_train.pkl'))
            pd.concat([X_val, y_val], axis=1).to_pickle(os.path.join(self.path, 'data_val.pkl'))

        save_thread = threading.Thread(target=save)
        save_thread.start()

    return wrapper

class SVM(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        
    @save_split_data
    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_cpus=None, sample_weight=None, **kwargs):
        time_start = time.time()
        X = self.preprocess(X)
        params = self._get_model_params()
        if "n_jobs" in params:
            params.pop('n_jobs')
            logger.log(15, "Multi-core not yet supported for SVMModel, this model will ignore them in training.")
        if num_cpus is not None:
            logger.log(15, "Multi-core not yet supported for SVMModel, this model will ignore them in training.")

        if sample_weight is not None: 
            logger.log(15, "sample_weight not yet supported for SVMModel, this model will ignore them in training.")

        num_rows_max = len(X)
        # FIXME: v0.1 Must store final num rows for refit_full or else will use everything! Worst case refit_full could train far longer than the original model.
        if time_limit is None or num_rows_max <= 10000:
            if self.problem_type in ['regression', 'softclass']:
                from sklearn.svm import SVR
                self.model = SVR(**params).fit(X, y)
            else:
                from sklearn.svm import SVC
                self.model = SVC(**params, probability=True).fit(X, y)
        else:
            self.model = self._fit_with_samples(X=X, y=y, model_params=params, time_limit=time_limit - (time.time() - time_start))

    def _fit_with_samples(self, X, y, model_params, time_limit, start_samples=10000, max_samples=None, sample_growth_factor=2, sample_time_growth_factor=8):

        time_start = time.time()

        num_rows_samples = []
        if max_samples is None:
            num_rows_max = len(X)
        else:
            num_rows_max = min(len(X), max_samples)
        num_rows_cur = start_samples
        while True:
            num_rows_cur = min(num_rows_cur, num_rows_max)
            num_rows_samples.append(num_rows_cur)
            if num_rows_cur == num_rows_max:
                break
            num_rows_cur *= sample_growth_factor
            num_rows_cur = math.ceil(num_rows_cur)
            if num_rows_cur * 1.5 >= num_rows_max:
                num_rows_cur = num_rows_max

        def sample_func(chunk, frac):
            # Guarantee at least 1 sample (otherwise log_loss would crash or model would return different column counts in pred_proba)
            n = max(math.ceil(len(chunk) * frac), 1)
            return chunk.sample(n=n, replace=False, random_state=0)

        if self.problem_type != 'regression':
            y_df = y.to_frame(name="label").reset_index(drop=True)
        else:
            y_df = None

        time_start_sample_loop = time.time()
        time_limit_left = time_limit - (time_start_sample_loop - time_start)
        model_type = self._get_model_type()
        idx = None
        for i, samples in enumerate(num_rows_samples):
            if samples != num_rows_max:
                if self.problem_type == 'regression':
                    idx = np.random.choice(num_rows_max, size=samples, replace=False)
                else:
                    idx = y_df.groupby("label", group_keys=False).apply(sample_func, frac=samples / num_rows_max).index
                X_samp = X[idx, :]
                y_samp = y.iloc[idx]
            else:
                X_samp = X
                y_samp = y
                idx = None
            self.model = model_type(**model_params).fit(X_samp, y_samp)
            time_limit_left_prior = time_limit_left
            time_fit_end_sample = time.time()
            time_limit_left = time_limit - (time_fit_end_sample - time_start)
            time_fit_sample = time_limit_left_prior - time_limit_left
            time_required_for_next = time_fit_sample * sample_time_growth_factor
            logger.log(15, f"\t{round(time_fit_sample, 2)}s \t= Train Time (Using {samples}/{num_rows_max} rows) ({round(time_limit_left, 2)}s remaining time)")
            if time_required_for_next > time_limit_left and i != len(num_rows_samples) - 1:
                logger.log(
                    20,
                    f"\tNot enough time to train SVM model on all training rows. Fit {samples}/{num_rows_max} rows. (Training KNN model on {num_rows_samples[i+1]} rows is expected to take {round(time_required_for_next, 2)}s)",
                )
                break
        if idx is not None:
            idx = set(idx)
            self._X_unused_index = [i for i in range(num_rows_max) if i not in idx]
        return self.model
    
    def _set_default_params(self):
        default_params = {
            
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
    
    def _get_default_searchspace(self) -> dict:
        spaces = {
        }
        return spaces
    
    
class XGB(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ohe: bool = True
        self._ohe_generator = None
        self._xgb_model_type = None

    def _set_default_params(self):
        default_params = {  
            
        }
        return default_params
    
    def _get_default_searchspace(self):
        return super()._get_default_searchspace()
    
    def _get_model_type(self):
        if self.problem_type in ['regression', 'softclass']:
            from xgboost import XGBRegressor
            return XGBRegressor
        else:
            from xgboost import XGBClassifier
            return XGBClassifier
    
    @save_split_data
    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_gpus=0, num_cpus=None, sample_weight=None, sample_weight_val=None, verbosity=2, **kwargs):
        # TODO: utilize sample_weight_val in early-stopping if provided
        start_time = time.time()
        ag_params = self._get_ag_params()
        params = self._get_model_params()
        if num_cpus:
            params["n_jobs"] = num_cpus
        max_category_levels = params.pop("proc.max_category_levels", 100)
        enable_categorical = params.get("enable_categorical", False)
        if enable_categorical:
            """Skip one-hot-encoding and pass categoricals directly to XGBoost"""
            self._ohe = False
        else:
            """One-hot-encode categorical features"""
            self._ohe = True

        X = self.preprocess(X, is_train=True, max_category_levels=max_category_levels)
        num_rows_train = X.shape[0]

        if X_val is None:
            early_stopping_rounds = None
            eval_set = None
        else:
            X_val = self.preprocess(X_val, is_train=False)
            eval_set.append((X_val, y_val))
            early_stopping_rounds = ag_params.get("early_stop", "adaptive")
            if isinstance(early_stopping_rounds, (str, tuple, list)):
                early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=early_stopping_rounds)

        if num_gpus != 0:
            params["tree_method"] = "gpu_hist"
            if "gpu_id" not in params:
                params["gpu_id"] = 0
        elif "tree_method" not in params:
            params["tree_method"] = "hist"

        model_type = self._get_model_type()
        self.model = model_type(**params)
        self.model.fit(X=X, y=y, eval_set=eval_set, verbose=False, sample_weight=sample_weight)

        bst = self.model.get_booster()
        # TODO: Investigate speed-ups from GPU inference
        # bst.set_param({"predictor": "gpu_predictor"})

        self.params_trained["n_estimators"] = bst.best_ntree_limit
        # Don't save the callback or eval_metric objects
        self.model.set_params(callbacks=None, eval_metric=None)
    
    
class KNN(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_model_type(self):
        if self.problem_type in ['regression', 'softclass']:
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor
        else:
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier
        
    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X
    
    def _set_default_params(self):
        default_params = {
            
        }
        return default_params
    
    def _get_default_searchspace(self):
        spaces = {
            'n_neighbors' : space.Int(lower=3, upper=1000),
            'weights' : space.Categorical('uniform', 'distance'),
            'algorithm' : space.Categorical('auto', 'ball_tree', 'ball_tree', 'brute'),
            'p' : space.Categorical(1, 2)
        }
        return spaces
    
    @save_split_data
    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_cpus=None, sample_weight=None, **kwargs):
        time_start = time.time()
        X = self.preprocess(X)
        params = self._get_model_params()
        if "n_jobs" not in params:
            params["n_jobs"] = num_cpus
        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for KNNModel, this model will ignore them in training.")

        num_rows_max = len(X)
        # FIXME: v0.1 Must store final num rows for refit_full or else will use everything! Worst case refit_full could train far longer than the original model.
        if time_limit is None or num_rows_max <= 10000:
            self.model = self._get_model_type()(**params).fit(X, y)
        else:
            self.model = self._fit_with_samples(X=X, y=y, model_params=params, time_limit=time_limit - (time.time() - time_start))

    def _fit_with_samples(self, X, y, model_params, time_limit, start_samples=10000, max_samples=None, sample_growth_factor=2, sample_time_growth_factor=8):
        time_start = time.time()

        num_rows_samples = []
        if max_samples is None:
            num_rows_max = len(X)
        else:
            num_rows_max = min(len(X), max_samples)
        num_rows_cur = start_samples
        while True:
            num_rows_cur = min(num_rows_cur, num_rows_max)
            num_rows_samples.append(num_rows_cur)
            if num_rows_cur == num_rows_max:
                break
            num_rows_cur *= sample_growth_factor
            num_rows_cur = math.ceil(num_rows_cur)
            if num_rows_cur * 1.5 >= num_rows_max:
                num_rows_cur = num_rows_max

        def sample_func(chunk, frac):
            # Guarantee at least 1 sample (otherwise log_loss would crash or model would return different column counts in pred_proba)
            n = max(math.ceil(len(chunk) * frac), 1)
            return chunk.sample(n=n, replace=False, random_state=0)

        if self.problem_type != 'regression':
            y_df = y.to_frame(name="label").reset_index(drop=True)
        else:
            y_df = None

        time_start_sample_loop = time.time()
        time_limit_left = time_limit - (time_start_sample_loop - time_start)
        model_type = self._get_model_type()
        idx = None
        for i, samples in enumerate(num_rows_samples):
            if samples != num_rows_max:
                if self.problem_type == 'regression':
                    idx = np.random.choice(num_rows_max, size=samples, replace=False)
                else:
                    idx = y_df.groupby("label", group_keys=False).apply(sample_func, frac=samples / num_rows_max).index
                X_samp = X[idx, :]
                y_samp = y.iloc[idx]
            else:
                X_samp = X
                y_samp = y
                idx = None
            self.model = model_type(**model_params).fit(X_samp, y_samp)
            time_limit_left_prior = time_limit_left
            time_fit_end_sample = time.time()
            time_limit_left = time_limit - (time_fit_end_sample - time_start)
            time_fit_sample = time_limit_left_prior - time_limit_left
            time_required_for_next = time_fit_sample * sample_time_growth_factor
            logger.log(15, f"\t{round(time_fit_sample, 2)}s \t= Train Time (Using {samples}/{num_rows_max} rows) ({round(time_limit_left, 2)}s remaining time)")
            if time_required_for_next > time_limit_left and i != len(num_rows_samples) - 1:
                logger.log(
                    20,
                    f"\tNot enough time to train KNN model on all training rows. Fit {samples}/{num_rows_max} rows. (Training KNN model on {num_rows_samples[i+1]} rows is expected to take {round(time_required_for_next, 2)}s)",
                )
                break
        if idx is not None:
            idx = set(idx)
            self._X_unused_index = [i for i in range(num_rows_max) if i not in idx]
        return self.model

class ANN(TabularNeuralNetTorchModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_default_params(self):
        default_params = {
            
        }
        return super()._set_default_params()
    
    def _get_default_searchspace(self):
        spaces = {
            "learning_rate": space.Real(1e-4, 3e-2, default=3e-4, log=True),
            "dropout_prob": space.Categorical(0.1, 0.0, 0.5, 0.2, 0.3, 0.4),
            "embedding_size_factor": space.Categorical(1.0, 0.5, 1.5, 0.7, 0.6, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4),
            "proc.embed_min_categories": space.Categorical(4, 3, 10, 100, 1000),
            "proc.impute_strategy": space.Categorical("median", "mean", "most_frequent"),
            "proc.max_category_levels": space.Categorical(100, 10, 20, 200, 300, 400, 500, 1000, 10000),
            "proc.skew_threshold": space.Categorical(0.99, 0.2, 0.3, 0.5, 0.8, 0.9, 0.999, 1.0, 10.0, 100.0),
            "use_batchnorm": space.Categorical(False, True),
            "num_layers": space.Categorical(2, 3, 4),
            "hidden_size": space.Categorical(128, 256, 512),
            "activation": space.Categorical("relu", "elu"),
            "weight_decay": space.Real(1e-12, 1.0, default=1e-6, log=True),
            "gamma": space.Real(0.1, 10.0, default=5.0),
            "alpha": space.Categorical(0.001, 0.01, 0.1, 1.0),
        }
        return spaces
    
    @save_split_data
    def _fit(self, X, y, X_val=None, y_val=None, **kwargs):
        super()._fit(X, y, **kwargs)