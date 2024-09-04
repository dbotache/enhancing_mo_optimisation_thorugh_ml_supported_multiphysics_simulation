import os, sys
import numpy as np
import pandas as pd
import ast

sys.path.append(os.path.abspath('..'))

from data_utils import load_df
from models.model_utils import ModelLoader, ModelWrapper
from explainability.feature_importance import get_feature_importance, plot_feature_importance
from explainability.pdp import pdp, plot_dpd

def explain_model(args):

    X_train, X_test, y_train, y_test = load_df(args)

    if args.xai.method == 'feature_relevance':

        print('Model Type: ', args.model.model_type)

        if args.model.model_type != 'xgb':
            print('xAI Error: Feature-Relevance only supported with XGB Models')

        else:
            xgb_object = ModelLoader(main_path=args.main_path,
                                     file_name=args.file_name,
                                     model_type=args.model.model_type,
                                     model_sub_folder=None,
                                     verbose=False)

            print('XGB loaded')

            feature_importance_df = get_feature_importance(X_test, y_test, xgb_object)

            if not os.path.isdir(os.path.join(args.main_path, "xai", args.file_name, args.splitfolder)):
                os.makedirs(os.path.join(args.main_path, "xai", args.file_name, args.splitfolder))

            feature_importance_df.to_hdf(f'{args.main_path}/xai/{args.file_name}/{args.splitfolder}/feature_importance.h5', key='score')
            feature_importance_df.to_csv(f'{args.main_path}/xai/{args.file_name}/{args.splitfolder}/feature_importance.csv')

            print(feature_importance_df)

            if args.xai.plot:

                save_path = f'{args.main_path}/xai/{args.file_name}/{args.splitfolder}'
                plot_feature_importance(feature_importance_df, type=args.xai.plot_type, save_path=save_path)

    if args.xai.method == 'partial_dependencies':

        wrapper = ModelWrapper(main_path=args.main_path,
                               file_name=args.file_name,
                               model_types=args.xai.model_types,
                               model_sub_folder=None,
                               verbose=False)

        if ast.literal_eval(args.xai.feature_name) is None:
            feature_name = X_test.columns.values[1]
            print(f'Plotting dependencies considering the Parameter: {feature_name}')
            feature_range, prediction_list = pdp(X_test, y_test, wrapper, feature_name)

        else:
            print(f'Plotting dependencies considering the Parameter: {args.xai.feature_name}')
            feature_range, prediction_list = pdp(X_test, y_test, wrapper, args.xai.feature_name)

        if not os.path.isdir(os.path.join(args.main_path, "xai", args.file_name, args.splitfolder)):
            os.makedirs(os.path.join(args.main_path, "xai", args.file_name, args.splitfolder))

        save_path = f'{args.main_path}/xai/{args.file_name}/{args.splitfolder}'
        plot_dpd(feature_range, prediction_list, y_test.columns,
                 args.xai.model_types, save_path=save_path)