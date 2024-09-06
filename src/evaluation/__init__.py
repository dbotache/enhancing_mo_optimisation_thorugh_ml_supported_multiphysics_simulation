import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from evaluation.pareto_front import return_opt_sample_pred_lists, pareto_front_2d, plot_front_2d

sys.path.append(os.path.abspath('..'))

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

def save_pareto_front(args):
    _, _, _, y_test = load_df(args)
    sample_list, pred_list = return_opt_sample_pred_lists(args)

    if args.file_name == "cfd_red":
        loc_target_x, loc_target_y, loc_target_z = "pressure_loss", "cooling_power", None
        maximize_y = False

    elif args.file_name == "Beispiel_Maschine_KI_Databasis":
        loc_target_x, loc_target_y, loc_target_z = "P_loss_total", "M", "Masse_mag"
        maximize_y = True

    front_list = []
    for i, loc_pred in enumerate(pred_list):
        pred = loc_pred['predictions']
        model_type = loc_pred['model_type']
        loc_front = pareto_front_2d(pred, loc_target_x, loc_target_y, maximize_y=maximize_y)
        front_list.append(loc_front)

        loc_df_list = [pred, y_test]
        loc_labels = [model_type, 'Databasis']

        figsize = (5, 4)
        colors = ['steelblue', 'red', 'darkred', 'mediumpurple', 'blue', 'forestgreen']

        plot_front_2d(loc_front, loc_target_x, loc_target_y, df_list=loc_df_list, labels=loc_labels,
                      colors=loc_target_z, color_group=colors,
                      alpha_list=[1, .5, .1, .1, .1], dot_size=10, figsize=figsize)

        plt.title(model_type)
        plt.legend(loc='upper left', bbox_to_anchor=(-.5, -.2), ncol=3)
        plt.tight_layout()
        plt.savefig(f'{args.main_path}/evaluation/{args.file_name}/{args.splitfolder}/pareto_predictions_{model_type}.svg')
        plt.savefig(f'{args.main_path}/evaluation/{args.file_name}/{args.splitfolder}/pareto_predictions_{model_type}.pdf')

    

