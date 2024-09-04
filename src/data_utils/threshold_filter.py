import numpy as np


def filter_data(X, bounds, verbose=False):
    """
    Takes a Dataframe [X] and a list of bounds [dict or list] and
    returns a list of index to drop from Dataframe

    single bounds:
    --------------

    bounds = {
    "col_names" : ['col 1', 'col 2'],
    "bounds" : [0, np.inf]
    }

    multiple bounds example:
    ------------------------

    bound_1 = {
    "col_names" : ['col 1', 'col 2'],
    "bounds" : [0, np.inf]
    }

    bound_2 = {
    "col_names" : ['col 3', 'col 4'],
    "bounds" : [0, np.inf]
    }

    bounds = [bound_1, bound_2]

    """

    index_to_drop = None
    if type(bounds) == list:

        loc_list = []
        for bound_i in bounds:
            loc_list.append(get_index_to_drop(X, bound_i))

        index_to_drop = np.unique(np.hstack(loc_list))

    elif type(bounds) == dict:
        index_to_drop = get_index_to_drop(X, bounds)

    if verbose:
        print("dropped index: ", index_to_drop)

    return index_to_drop


def get_index_to_drop(X, bounds):
    # check if column names available in Dataframe
    assert [i for i in bounds["col_names"] if i in X.columns.values] == bounds["col_names"]

    index_to_drop = []

    for col_name in bounds["col_names"]:
        # Extract the lower and upper bounds for the current column
        lower_bound, upper_bound = bounds["bounds"]

        # Identify the indices that satisfy the conditions
        bool_selection = (X[col_name] < lower_bound) | (X[col_name] > upper_bound)

        # Append the indices to the list
        index_to_drop.append(X[bool_selection].index.values)

    loc_list = np.unique(np.hstack(index_to_drop))

    return loc_list
