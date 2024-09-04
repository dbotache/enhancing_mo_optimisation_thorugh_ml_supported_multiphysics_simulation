import numpy as np
import pandas as pd

def pdp(X, y, model_wrapper, feature, model_types=None):
    if model_types == None:
        model_types = ['xgb', 'ensemble', 'mlp', 'cnn']

    # get min and max of feature range
    feature_idx = X.columns.get_loc(feature)
    min, max = X.iloc[:, feature_idx].min(), X.iloc[:, feature_idx].max()
    feature_range = np.linspace(min, max, 100)

    # compute avg of other features
    avgs = []
    for feat in range(len(X.columns)):
        avgs.append(X.iloc[:, feat].mean())

    preds = [list() for i in model_types]
    for val in feature_range:
        # generate input sample and set other input features to avg
        x_ = np.array(avgs)
        x_[feature_idx] = val

        x_df = pd.DataFrame(x_.reshape(1, -1), columns=X.columns)

        prediction_list = model_wrapper.return_predictions(x_df)

        for i, model_type in enumerate(model_types):
            preds[i].append(prediction_list[i])

    return feature_range, preds