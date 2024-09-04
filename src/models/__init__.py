import warnings
import argparse

warnings.filterwarnings('ignore')

from data_utils import load_df
from data_utils.scaling import FeatureTargetScaling
from models.xgb_optimisation import optimize_xgb, train_xgb
from models.mlp_optimisation import optimize_mlp, train_mlp
from models.cnn_optimisation import optimize_cnn, train_cnn
from models.ensemble import train_ensembles
from ressources import n_cpu, device


def model_optimisation(args):

    X_train, X_test, y_train, y_test = load_df(args)

    x_y_scaling = FeatureTargetScaling(X_train, y_train,
                                       scaler_type_features=args.model.feature_scaler,
                                       scaler_type_targets=args.model.target_scaler)

    X_train, y_train = x_y_scaling.scale_data(X_train, y_train)
    X_test, y_test = x_y_scaling.scale_data(X_test, y_test)

    if args.model.model_type == 'xgb':
        optimize_xgb(args, device, X_train, y_train,
                     n_cpu=n_cpu, trials=args.model.opt_trials, metric=args.model.metric)
    if args.model.model_type == 'ensemble': # Hyperparameter tuning is done automatically
        train_ensembles(args, X_train, y_train,
                        n_cpu=n_cpu, metric=args.model.metric)
    if args.model.model_type == 'mlp':
        optimize_mlp(args, X_train, y_train, X_test, y_test, device, cv_n_splits=5,
                     n_cpu=n_cpu, trials=args.model.opt_trials, metric=args.model.metric)
    if args.model.model_type == 'cnn':
        optimize_cnn(args, X_train, y_train, X_test, y_test, device, cv_n_splits=5,
                     n_cpu=n_cpu, trials=args.model.opt_trials, metric=args.model.metric)

def model_train(args):

    X_train, X_test, y_train, y_test = load_df(args)

    x_y_scaling = FeatureTargetScaling(X_train, y_train,
                                       scaler_type_features=args.model.feature_scaler,
                                       scaler_type_targets=args.model.target_scaler)

    X_train, y_train = x_y_scaling.scale_data(X_train, y_train)
    X_test, y_test = x_y_scaling.scale_data(X_test, y_test)

    if args.model.model_type == 'xgb':
        train_xgb(args, device, X_train, y_train,
                  n_cpu=n_cpu, metric=args.model.metric)
    if args.model.model_type == 'ensemble': # Hyperparameter tuning is done automatically
        train_ensembles(args, X_train, y_train,
                        n_cpu=n_cpu, metric=args.model.metric)
    if args.model.model_type == 'mlp':
        train_mlp(args, X_train, y_train, X_test, y_test, device,
                  n_cpu=n_cpu, metric=args.model.metric)
    if args.model.model_type == 'cnn':
        train_cnn(args, X_train, y_train, X_test, y_test, device,
                  n_cpu=n_cpu, metric=args.model.metric)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Hyper-Parameter Optimisation of multiple models',
                                     allow_abbrev=False)
    parser.add_argument('--data_path', metavar='data_path', type=str,
                        help='Data_Path where DataFrames are located', default='../data')
    parser.add_argument('--save_folder', metavar='save_folder', type=str,
                        help='Folder destination for saving models', default='../data/models')
    parser.add_argument('--file_name', metavar='file_name', type=str,
                        help='Name of Datafile', default='DoE_29V_791D')
    parser.add_argument('--model_type', metavar='model_type', type=str,
                        help='Provide only a single model_type for optimisation',
                        default='ensemble')
    parser.add_argument('--feature_scaler', metavar='feature_scaler', type=str,
                        help='Scaling strategy for Features [standard or minmax]', default=None)
    parser.add_argument('--target_scaler', metavar='target_scaler', type=str,
                        help='Provide scaling strategy for targets if necessary [standard or minmax]', default=None)
    parser.add_argument('--num_epochs', metavar='num_epochs', type=int,
                        help='Number of epochs for training MLP or CNN', default=10)
    parser.add_argument('--batch_size', metavar='batch_size', type=int,
                        help='Batch_size in training loop of MLP or CNN', default=16)
    parser.add_argument('--opt_trials', metavar='opt_trials', type=int,
                        help='Number of trials in hyperparameter optimisation loop', default=50)
    parser.add_argument('--metric', metavar='metric', type=str,
                        help='Metric to use in training', default='MSE')

    args = parser.parse_args()

    print(args)
    model_optimisation(args)
