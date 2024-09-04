import pandas as pd
from pymoo.core.variable import Real, Integer

from data_utils.feature_constraints import get_min_max_features
from data_utils.threshold_filter import filter_data

from opt.opt_utils import MyOutput_cfd, MyOutput_motor
from opt.opt_utils import opt_fn_cfd, opt_fn_motor, opt_fn_motor_free
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
        X = pd.read_hdf(f"{args.main_path}/Dataframes/{args.splitfolder}/X_test_{args.file_name}.h5", key="features")
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
        X = pd.read_hdf(f"{args.main_path}/Dataframes/{args.splitfolder}/X_test_{args.file_name}.h5", key="features")
        var_names = X.columns

        x_drop_index = filter_data(X, CONSTRAINTS_FEATURES_BM)
        X = X.drop(index=x_drop_index)
        #print('Droped index : ', x_drop_index)
        #print(X.max(), X.min())

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
            # "MyOutput": MyOutput_cfd, # TODO: Why do we use here the Class MyOutput_cfd?
            "MyOutput": MyOutput_motor,
        }
        return optimization_dict

    elif "1Q_1V" in args.file_name:
        X = pd.read_hdf(f"{args.main_path}/Dataframes/{args.splitfolder}/X_test_{args.file_name}.h5", key="features")
        var_names = X.columns

        ranges_list = CONSTRAINTS_FEATURES_1Q_1V_BASE_PLUS_OPT # Ranges around the reference
        print('Using Base Plus Range Constraints')

        variables = dict()
        for var_name in var_names:

            bool_empty = True
            for r_ in ranges_list:
                if r_['col_names'][0] == var_name:
                    min_ = r_['bounds'][0]
                    max_ = r_['bounds'][1]
                    variables[var_name] = Real(bounds=(min_, max_))
                    print(f'Setting custom range for {var_name} min:{min_} max:{max_}')
                    bool_empty = False
            if bool_empty:
                min_ = X.loc[:, [var_name]].min().item()
                max_ = X.loc[:, [var_name]].max().item()
                variables[var_name] = Real(bounds=(min_, max_))
                print(f'Setting range based on Data for {var_name} min:{min_} max:{max_}')

            elif var_name == "vdp_w_sp" and bool_empty:
                variables[var_name] = Integer(bounds=(X.loc[:, [var_name]].min().item(),
                                                      X.loc[:, [var_name]].max().item()))

        optimization_dict = {
            "target_str_list": [
                "M(S2,n1)",
                "M(S2,n2)",
                "M(S2,n3)",
                "M(S2,n4)",
                "Pv_Antrieb_Fzg_Zykl_1",
                "Pv_Antrieb_Fzg_Zykl_2",
                "MEK_Aktivteile",
            ],
            "n_obj": 3,
            "opt_fn": opt_fn_motor,
            "variables": variables,
            "var_names": var_names,
            "MyOutput": MyOutput_motor,
        }
        return optimization_dict

    elif args.file_name != "Beispiel_Maschine_KI_Databasis" and args.file_name !=  "cfd_red":
        X = pd.read_hdf(f"{args.main_path}/Dataframes/{args.splitfolder}/X_test_{args.file_name}.h5", key="features")
        var_names = X.columns
        n_var = len(var_names)
        xl, xu = get_min_max_features(X)

        variables = dict()
        for i in range(len(var_names)):
            if var_names[i] == "vdp_w_sp":
                variables[var_names[i]] = Integer(bounds=(xl[i], xu[i]))
            else:
                variables[var_names[i]] = Real(bounds=(xl[i], xu[i]))

        optimization_dict = {
            "target_str_list": [
                "M(S2,n1)",
                "M(S2,n2)",
                "M(S2,n3)",
                "M(S2,n4)",
                "Pv_Antrieb_Fzg_Zykl_1",
                "Pv_Antrieb_Fzg_Zykl_2",
                "MEK_Aktivteile",
            ],
            "n_obj": 3,
            "opt_fn": opt_fn_motor,
            "variables": variables,
            "var_names": var_names,
            "MyOutput": MyOutput_motor,
        }
        return optimization_dict

    else:
        print("not implemented")
        optimization_dict = {}
        return optimization_dict

