import numpy as np

"""
    Including feature constraints using a List of Dictionaries with the corresponding values
    Only if the value-range of a specific parameter needs to be adjusted
"""
CONSTRAINTS_FEATURES_EXAMPLE = [
    {"col_names": ["x1"], "bounds": [1.8, 2.2]},
    {"col_names": ["x2"], "bounds": [5, 5.8]},
    {"col_names": ["x3"], "bounds": [21, 25]},
    {"col_names": ["x4", "x5", "x6"], "bounds": [14, 16.5]}
]


CONSTRAINTS_FEATURES_BM = [
    {"col_names": ["wi_ac1"], "bounds": [1.8, 2.2]},
    {"col_names": ["wi_ac2"], "bounds": [5.4, 5.8]},
    {"col_names": ["hi_mag1"], "bounds": [21.3, 24]},
    {"col_names": ["hi_mag2"], "bounds": [14.6, 16]}
]

def get_feature_bounds(X, constraints=None):
    """
    Function to create a list of dictionaries with range-values

    :param X: pd.DataFrame
    :param constraints: dict() - Additional Constraints defined on advance
    :return: List of dictionaries
    """
    constraint_cols = []
    if constraints is not None:

        loc_constraints = constraints.copy()

        for i in loc_constraints:
            constraint_cols = constraint_cols + i['col_names']
    else:
        loc_constraints = []

    for col in X.columns.values:

        if col in constraint_cols:
            pass
        else:
            loc_bound = {
                "col_names": [col],
                "bounds": [X[col].min(), X[col].max()]
            }
            loc_constraints.append(loc_bound)

    return loc_constraints


def get_min_max_features(df):

    min_vals = []
    max_vals = []

    for column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        min_vals.append(min_val)
        max_vals.append(max_val)

    return np.array(min_vals), np.array(max_vals)