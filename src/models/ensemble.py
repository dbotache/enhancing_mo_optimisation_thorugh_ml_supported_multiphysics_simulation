import autosklearn
import autosklearn.regression
import pickle
import os


def train_ensembles(args, X_train, y_train, n_cpu=1, metric='MSE'):
    sub_folder = args['date']

    metric_object = None

    if metric == 'MSE':
        metric_object = autosklearn.metrics.mean_squared_error
    elif metric == 'MAE':
        metric_object = autosklearn.metrics.mean_absolute_error

    if not os.path.isdir(os.path.join(args.main_path, "models", args.model.model_type, args.file_name, sub_folder)):
        os.makedirs(os.path.join(args.main_path, "models", args.model.model_type, args.file_name, sub_folder))

    for target_name in y_train.columns.values:

        if target_name == 'A_chip_mm_sqare':
            continue

        loc_y_train = y_train.loc[:, [target_name]]

        if os.path.isdir(f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/{sub_folder}/{target_name}'):
            print(f'Model for target {target_name} already exist')

        else:
            print(f'Training ensemble on {target_name}')
            automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=240,
                memory_limit=16800,
                per_run_time_limit=60,
                delete_tmp_folder_after_terminate=False,
                tmp_folder=f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/{sub_folder}/{target_name}',
                n_jobs=n_cpu,
                metric=metric_object
            )
            automl.fit(X_train, loc_y_train, dataset_name=target_name)

            # save model
            with open(f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/{sub_folder}/{target_name}/ensemble.pkl', "wb") as f:
                pickle.dump(automl, f)
