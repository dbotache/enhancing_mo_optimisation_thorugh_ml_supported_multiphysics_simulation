import pandas as pd
from pymoo.core.variable import Real, Integer

from data_utils.feature_constraints import get_min_max_features
from data_utils.feature_constraints import CONSTRAINTS_FEATURES_BM
from data_utils.threshold_filter import filter_data
from data_utils import load_df

from opt.opt_motor import opt_fn_motor_free, MyOutput_motor
from opt.opt_cfd import opt_fn_cfd, MyOutput_cfd
from opt.ea_opt import ea_optimization


def make_opt(args):
    if args.opt.opt_type == "ea":
        problem_dict = build_ea_problem(args)
        ea_optimization(args, problem_dict)
    elif args.opt.opt_type == "gb":
        print("tbd")
    elif args.opt.opt_type == "rand":
        print("tbd")
    else:
        print("not implemented")


def build_ea_problem(args):
    if args.file_name == "cfd_red":
        _, X, _, _ = load_df(args)
        var_names = X.columns
        n_var = len(var_names)
        xl, xu = get_min_max_features(X)

        variables = dict()
        for i in range(len(var_names)):
            variables[var_names[i]] = Real(bounds=(xl[i], xu[i]))

        optimization_dict = {
            "target_str_list": ["pressure_loss", "cooling_power"],
            "n_obj": 2,
            "opt_fn": opt_fn_cfd,
            "variables": variables,
            "var_names": var_names,
            "MyOutput": MyOutput_cfd,
        }
        return optimization_dict

    elif args.file_name == "Beispiel_Maschine_KI_Databasis":
        _, X, _, _ = load_df(args)
        var_names = X.columns

        x_drop_index = filter_data(X, CONSTRAINTS_FEATURES_BM)
        X = X.drop(index=x_drop_index)
        xl, xu = get_min_max_features(X)

        variables = dict()
        for i in range(len(var_names)):
            variables[var_names[i]] = Real(bounds=(xl[i], xu[i]))

        optimization_dict = {
            "target_str_list": ["M", "P_loss_total", "Masse_mag"],
            "n_obj": 3,
            "opt_fn": opt_fn_motor_free,
            "variables": variables,
            "var_names": var_names,
            "MyOutput": MyOutput_motor,
        }
        return optimization_dict

    else:
        print("Optimisation Function not implemented")
        optimization_dict = {}
        return optimization_dict

