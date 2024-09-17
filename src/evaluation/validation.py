import os
import pandas as pd
from models.model_utils import ModelWrapper
from data_utils import load_df
from evaluation.model_evaluation import score_table


def get_validation_scores(args):

    X_train, _, y_train, _ = load_df(args)

    X_cols = X_train.columns.values
    y_cols = y_train.columns.values

    model_types = args.eval.model_types
    opt_alg = args.opt.opt_type

    if args.eval.validation_data_path == 'None':
        validation_path = os.path.normpath(os.getcwd())

        if not os.path.isdir(validation_path):
            print(f'Path does not exist: {validation_path}')
            return [], []
    else:
        validation_path = args.eval.validation_data_path

    load_path = f'{validation_path}/{args.file_name}/{args.splitfolder}/{opt_alg}'

    # use ModelWrapper Class
    wrapper = ModelWrapper(main_path=args.main_path,
                           file_name=args.file_name,
                           model_types=model_types,
                           model_sub_folder=None,
                           verbose=False)
    score_tables = []
    for i, model_type in enumerate(model_types):
        file_path = os.path.join(load_path, model_type, 'validation.h5')

        if not os.path.isfile(file_path):
            print(f'Solution candidates for {model_type} not available')
            continue

        loc_validation_df = pd.read_hdf(f'{file_path}')
        X_loc_validation = loc_validation_df.loc[:, X_cols]
        y_loc_validation = loc_validation_df.loc[:, y_cols]
        loc_pred_list = wrapper.return_predictions(X_loc_validation)

        loc_scores_table = score_table(loc_pred_list, y_loc_validation, model_types)
        score_tables.append(loc_scores_table.loc[loc_scores_table.regressor == model_type])

    return pd.concat(score_tables)


