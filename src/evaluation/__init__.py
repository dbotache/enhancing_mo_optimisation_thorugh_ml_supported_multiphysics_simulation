import os, sys

sys.path.append(os.path.abspath('..'))

#from evaluation.pareto_front import pareto_front_2d, pareto_front_3d, plot_front_2d, plot_front_3d
from evaluation.model_evaluation import score_table
from data_utils import load_df
from models.model_utils import ModelWrapper


def return_scores(args):

    X_train, X_test, y_train, y_test = load_df(args)

    model_types = args.eval.model_types

    # use ModelWrapper Class
    wrapper = ModelWrapper(main_path=args.main_path,
                           file_name=args.file_name,
                           model_types=model_types,
                           model_sub_folder=None,
                           verbose=False)

    pred_list = wrapper.return_predictions(X_test)
    scores_df = score_table(pred_list, y_test, model_types)

    if not os.path.isdir(os.path.join(args.main_path, "evaluation", args.file_name, args.splitfolder)):
        os.makedirs(os.path.join(args.main_path, "evaluation", args.file_name, args.splitfolder))

    scores_df.to_hdf(f'{args.main_path}/evaluation/{args.file_name}/{args.splitfolder}/scores.h5', key='score')
    scores_df.to_csv(f'{args.main_path}/evaluation/{args.file_name}/{args.splitfolder}/scores.csv')

    return scores_df
