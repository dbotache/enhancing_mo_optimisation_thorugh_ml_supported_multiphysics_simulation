import os
import json
import pickle
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.plot.contour import plot_contour
from sklearn.model_selection import KFold
import warnings

import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE


xgb_parameters = [{'name': 'n_estimators', 'type': 'choice', 'is_ordered' : True,
                   'values' : [100, 500, 1000, 1500, 2000, 2500, 3000]},
                  {'name': 'eta', 'type': 'range', 'value_type' : 'float', 'bounds': [1e-7, 1], 'log_scale': True},
                  {'name': 'reg_lambda' , 'type': 'range', 'value_type' : 'float', 'bounds': [0, 0.99]},
                  {'name': 'max_depth', 'type': 'range', 'value_type' : 'int', 'bounds': [4, 20]}]


xgb_param_exte = [{'name': 'n_estimators', 'type': 'choice', 'is_ordered' : True,
                   'values' : [100, 500, 1000, 1500, 2000, 2500, 3000]},
                  {'name': 'eta', 'type': 'range', 'value_type' : 'float', 'bounds': [1e-7, 1], 'log_scale': True},
                  {'name': 'subsample', 'type': 'range', 'value_type' : 'float', 'bounds': [0.1, 0.9]},
                  {'name': 'colsample_bytree' , 'type': 'range', 'value_type' : 'float', 'bounds': [0.1, 0.9]},
                  {'name': 'reg_alpha' , 'type': 'range', 'value_type' : 'float', 'bounds': [0, 0.99]},
                  {'name': 'min_child_weight' , 'type': 'range', 'value_type' : 'float', 'bounds': [0, 100]},
                  {'name': 'gamma', 'type': 'range', 'value_type' : 'float', 'bounds': [0, 100]},
                  {'name': 'reg_lambda' , 'type': 'range', 'value_type' : 'float', 'bounds': [0, 0.99]},
                  {'name': 'max_depth', 'type': 'range', 'value_type' : 'int', 'bounds': [4, 20]}]


class XGBBayesOptimizer:
    def __init__(self,
                 X_train,
                 y_train,
                 model_parameters,
                 X_eval=None,
                 y_eval=None,
                 cv_n_splits=5,
                 shuffle=True,
                 total_trials=15,
                 minimize=True,
                 objective_name='MSE'):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.cv_n_splits = cv_n_splits
        self.shuffle=shuffle
        self.model_parameters = model_parameters
        self.total_trials = total_trials
        self.minimize = minimize
        self.objective_name = objective_name
        self.print_param_iter = None
        
        # experiment variables
        self.best_parameters = None
        self.values = None
        self.experiment = None
        self.model = None
        
    def run_bayes_search(self, n_jobs=1, gpu_id=None, print_param_iter=False):
        self.n_jobs = n_jobs
        self.gpu_id = gpu_id
        self.print_param_iter = print_param_iter

        self.best_parameters, self.values, self.experiment, self.model = optimize(
            parameters=self.model_parameters,
            evaluation_function= self.function_to_optimize,
            objective_name=self.objective_name,
            total_trials=self.total_trials, 
            minimize=self.minimize,
        )

        return self.best_parameters, self.model
            
    def function_to_optimize(self, model_parameters):

        objective = None
        eval_metric = None

        if self.objective_name == 'MSE':
            objective = 'reg:squarederror'
            eval_metric = MSE
        elif self.objective_name == 'MAE':
            objective = 'reg:absoluteerror' # Version 1.7.0
            eval_metric = MAE
        
        if self.print_param_iter:
            print(model_parameters)
        
        warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

        if self.gpu_id is None:
            tree_method = 'auto'
            gpu_id = None
        else:
            tree_method = 'gpu_hist'
            gpu_id = self.gpu_id

        # If not evaluation set given -> run CV
        if self.X_eval is None:
            
            kf = KFold(n_splits=self.cv_n_splits, shuffle=self.shuffle)
            scores_kfold = []
            
            for train_index, eval_index in kf.split(self.X_train):

                regressor = xgb.XGBRegressor(**model_parameters,
                                             n_jobs=self.n_jobs,
                                             tree_method=tree_method,
                                             gpu_id=gpu_id,
                                             objective=objective,
                                             eval_metric=eval_metric
                                             )

                X_train, X_eval = self.X_train[train_index], self.X_train[eval_index]
                y_train, y_eval = self.y_train[train_index], self.y_train[eval_index]

                regressor.fit(X_train, y_train)
                pred = regressor.predict(X_eval)
                avg_mse = MSE(y_eval, pred)
                scores_kfold.append(avg_mse)
            
            trial_avg_scores = np.mean(scores_kfold)

        # Use evaluation set
        else:

            regressor = xgb.XGBRegressor(**model_parameters,
                                         n_jobs=self.n_jobs,
                                         tree_method=tree_method,
                                         gpu_id=gpu_id,
                                         objective=objective,
                                         eval_metric=eval_metric
                                         )
            
            regressor.fit(self.X_train, self.y_train)
            pred = regressor.predict(self.X_eval)
            trial_avg_scores = MSE(self.y_eval, pred)

        return trial_avg_scores
    
    def plot_countour_helper(self, model, param_x, param_y):
        render(
            plot_contour(
            model=model,
            param_x=param_x,
            param_y=param_y,
            metric_name=self.objective_name,
            )
        )
        

def optimize_xgb(args, device, X_train, y_train, n_cpu=1, trials=10, metric='MSE'):
    gpu_id = None
    tree_method = 'auto'
    objective = None
    eval_metric = None

    if metric == 'MSE':
        objective = 'reg:squarederror'
        eval_metric = MSE
    elif metric == 'MAE':
        objective = 'reg:absoluteerror'
        eval_metric = MAE

    if device == 'cuda':
        gpu_id = 0
        tree_method = 'gpu_hist'

    if not os.path.isdir(os.path.join(args.main_path, "models", args.model.model_type, args.file_name)):
        os.makedirs(os.path.join(args.main_path, "models", args.model.model_type, args.file_name))

    param_list = []
    targets_list = []

    for i in np.arange(y_train.shape[1]):

        target_name = y_train.columns.values[i]
        targets_list.append(target_name)

        print(f'Target value {i + 1} from {y_train.shape[1]}: {target_name}')

        optimizer = XGBBayesOptimizer(X_train=X_train.values,
                                      y_train=y_train.iloc[:, i].values,
                                      model_parameters=xgb_parameters,
                                      total_trials=trials,
                                      minimize=True,
                                      objective_name=metric)
        best_parameters, model = optimizer.run_bayes_search(n_jobs=n_cpu, gpu_id=gpu_id, print_param_iter=False)
        param_list.append(best_parameters)

        best_parameters['feature_scaler'] = args.model.feature_scaler
        best_parameters['target_scaler'] = args.model.target_scaler

        with open(f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/{target_name}_parameters.json', "w") as f:
            json.dump(best_parameters, f)

    print('XGB Optimization finished.')

    regressors = []

    for i, param_ in enumerate(param_list):
        print(f'Training xgb for target {y_train.columns.values[i]}')
        regressors.append(xgb.XGBRegressor(**param_,
                                           n_jobs=n_cpu,
                                           tree_method=tree_method,
                                           gpu_id=gpu_id,
                                           objective=objective,
                                           eval_metric=eval_metric))
        regressors[i].fit(X_train.values, y_train.values[:, i])

    # Save
    for i, reg_ in enumerate(regressors):
        pickle.dump(reg_,
                    open(f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/{targets_list[i]}.pkl', "wb"))


def train_xgb(args, device, X_train, y_train, n_cpu=1, metric='MSE'):
    gpu_id = None
    tree_method = 'auto'
    objective = None
    eval_metric = None

    if metric == 'MSE':
        objective = 'reg:squarederror'
        eval_metric = MSE
    elif metric == 'MAE':
        objective = 'reg:absoluteerror'
        eval_metric = MAE

    if device == 'cuda':
        gpu_id = 0
        tree_method = 'gpu_hist'

    if not os.path.isdir(os.path.join(args.main_path, "models", args.model.model_type, args.file_name)):
        print('Please create a folder and paste all config files for the '
              'corresponding Hyperparameters for each target value')

    for i in np.arange(y_train.shape[1]):

        loc_parameters = None

        target_name = y_train.columns.values[i]

        config_file_path = f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/{target_name}_parameters.json'

        if os.path.isfile(config_file_path):
            with open(config_file_path, "r") as f:
                loc_parameters = json.load(f)
        else:
            print('Please copy and paste all config files for the corresponding Hyperparameters for each target value')

        loc_parameters['feature_scaler'] = args.model.feature_scaler
        loc_parameters['target_scaler'] = args.model.target_scaler

        print(f'Training xgb for target {y_train.columns.values[i]}')
        print(f'Hyperparameters:  {loc_parameters}')
        reg_ = xgb.XGBRegressor(**loc_parameters,
                                n_jobs=n_cpu,
                                tree_method=tree_method,
                                gpu_id=gpu_id,
                                objective=objective,
                                eval_metric=eval_metric)
        reg_.fit(X_train.values, y_train.values[:, i])

        pickle.dump(reg_,
                    open(f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/{target_name}.pkl',
                         "wb"))
