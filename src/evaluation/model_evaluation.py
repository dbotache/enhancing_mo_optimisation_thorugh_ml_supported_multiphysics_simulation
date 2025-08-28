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


def score_table(pred_list, y_true, model_types, rename_target_dict=None, sort_by='target'):

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

    scores_table = scores_df.loc[:, scores_df.columns.values[::-1]].sort_values(by=sort_by).reset_index().drop(
        columns='index').set_index('target')

    return scores_table


def compare_model_scores(y_true, predictions, labels=None):
    """
    Display regression metrics for one or multiple prediction sets.

    Parameters:
    ----------
    y_true : array-like
        True target values.
    predictions : array-like, DataFrame, or list/tuple of them
        Predictions to evaluate. Can be a single set or multiple sets.
    labels : list/tuple of str, optional
        Column labels for the prediction sets.
    """
    # Ensure predictions is a list
    if not isinstance(predictions, (list, tuple)):
        predictions = [predictions]

    # Convert y_true to numpy array
    y_true = np.asarray(y_true)

    # Default labels if not provided
    if labels is None:
        labels = [f"Pred {i + 1}" for i in range(len(predictions))]

    def bias_score(y_true, y_pred):
        return np.mean(y_pred - y_true)

    def nbias_score(y_true, y_pred):
        denom = np.max(np.abs(y_true))
        return np.mean(y_pred - y_true) / denom if denom != 0 else np.nan

    def rmse_score(y_true, y_pred):
        return np.sqrt(MSE(y_true, y_pred))

    def nrmse_score(y_true, y_pred):
        denom = np.max(np.abs(y_true))
        return rmse_score(y_true, y_pred) / denom if denom != 0 else np.nan

    def nmae_score(y_true, y_pred):
        denom = np.max(np.abs(y_true))
        return MAE(y_true, y_pred) / denom if denom != 0 else np.nan

    def nmse_score(y_true, y_pred):
        denom = np.var(y_true)
        return MSE(y_true, y_pred) / denom if denom != 0 else np.nan

    metrics = {
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": rmse_score,
        "NMAE": nmae_score,
        "NMSE": nmse_score,
        "NRMSE": nrmse_score,
        "Bias": bias_score,
        "NBias": nbias_score,
        "R2": r2_score,
        "MAPE [%]": MAPE
    }

    results = []
    for name, func in metrics.items():
        row = [name]
        for pred in predictions:
            pred = np.asarray(pred)
            val = func(y_true, pred)
            if name == "MAPE":
                row.append(f"{val * 100:.4f}")
            else:
                row.append(f"{val:.4f}")
        results.append(row)

    df = pd.DataFrame(results, columns=["Metric"] + labels)
    scores_df = pd.DataFrame(df.T.values[1:], columns=df.T.iloc[[0]].values.flatten(), index=df.columns.values[1:]).astype(float)
    return scores_df