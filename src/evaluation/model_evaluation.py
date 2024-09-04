from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd


def evaluate_reg(y_true, y_pred):
    """
    Calculates several regression scores between the true and predicted values.

    Args:
        y_true (pd.DataFrame or np.array): Ground truth target values.
        y_pred (pd.DataFrame or np.array): Predicted target values.

    Returns:
        dict: A dictionary with the following regression scores:
              "bias": Bias between the true and predicted values.
              "rmse": Root mean squared error.
              "mape": Mean absolute percentage error.
              "nbias": Normalized bias (bias divided by the mean of the true values).
              "mae": Mean absolute error.
              "nmae": Normalized mean absolute error (MAE divided by the mean of the true values).
              "mse": Mean squared error.
              "nmse": Normalized mean squared error (MSE divided by the variance of the true values).
              "nrmse": Normalized root mean squared error (RMSE divided by the standard deviation of the true values).
              "r2": R^2 score.
    """

    # Convert input data to numpy arrays if necessary
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    # cols need to be numeric
    y_pred = np.array(y_pred.astype(float))
    y_true = np.array(y_true.astype(float))

    # Calculate regression scores
    bias = np.mean(y_pred - y_true)
    rmse = np.sqrt(MSE(y_true, y_pred))
    mape = MAPE(y_true=y_true, y_pred=y_pred)
    nbias = bias / np.mean(y_true)
    mae = MAE(y_true, y_pred)
    nmae = mae / np.mean(y_true)
    mse = MSE(y_true, y_pred)
    nmse = mse / np.var(y_true, ddof=1)
    nrmse = rmse / np.std(y_true, ddof=1)
    r2 = r2_score(y_true, y_pred)

    # Return results as a dictionary
    return {
        "bias": bias,
        "rmse": rmse,
        "mape": mape,
        "nbias": nbias,
        "mae": mae,
        "nmae": nmae,
        "mse": mse,
        "nmse": nmse,
        "nrmse": nrmse,
        "r2": r2,
    }


def score_table(pred_list, y_true, model_types, rename_target_dict=None):

    if rename_target_dict is not None:
        y_true = y_true.rename(columns=rename_target_dict)

        for i, pred in enumerate(pred_list):
            pred_list[i] = pred.rename(columns=rename_target_dict)

    df_list = []

    for i, model_type in enumerate(model_types):
        scores_list = []
        for col in y_true.columns.values:
            loc_scores = evaluate_reg(y_true.loc[:, [col]].values, pred_list[i].loc[:, [col]].values)
            loc_df = pd.DataFrame.from_dict(loc_scores, orient='index').transpose()
            loc_df['target'] = col
            loc_df['regressor'] = f'{model_type}'
            scores_list.append(loc_df)

        df_list.append(pd.concat(scores_list))

    scores_df = pd.concat(df_list)

    scores_table = scores_df.loc[:, scores_df.columns.values[::-1]].sort_values(by='target').reset_index().drop(
        columns='index').set_index('target')

    return scores_table