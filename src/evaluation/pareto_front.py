import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from models.model_utils import ModelWrapper


def pareto_front_2d(df, x_column, y_column, maximize_y=True):
    """
    Plot a 2D Pareto front from a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - x_column (str): The column name for the x-axis values.
    - y_column (str): The column name for the y-axis values.
    - maximize_y (bool): True if y should be maximized, False if minimized.

    Returns:
    - None (displays the plot).
    """
    # Check if the DataFrame is empty or lacks necessary columns
    if df.empty or x_column not in df.columns or y_column not in df.columns:
        print("Invalid DataFrame or columns.")
        return

    # Sort the DataFrame by x_column and y_column
    if maximize_y:
        df_sorted = df.sort_values(by=[x_column, y_column], ascending=[True, False])
    else:
        df_sorted = df.sort_values(by=[x_column, y_column], ascending=[True, True])

    # Initialize the Pareto front
    pareto_front = []
    if maximize_y:
        min_y = float('-inf')
    else:
        min_y = float('inf')

    # Iterate through the sorted DataFrame to find the Pareto front
    for index, row in df_sorted.iterrows():
        if maximize_y:
            if row[y_column] > min_y:
                pareto_front.append(row)
                min_y = row[y_column]
        else:
            if row[y_column] < min_y:
                pareto_front.append(row)
                min_y = row[y_column]

    return pd.DataFrame(pareto_front)


def plot_front_2d(pareto_front, x_column, y_column, df_list=None, front_label='Pareto Front', labels=None, colors=None,
                  color_group=None, alpha_list=None, figsize=(5, 4), dot_size=20):
    """
    Plot a 2D Pareto front and optionally overlay multiple DataFrames on the same plot.

    Parameters:
    - pareto_front (pd.DataFrame): The DataFrame containing the Pareto front data.
    - x_column (str): The column name for the x-axis values.
    - y_column (str): The column name for the y-axis values.
    - df_list (list of pd.DataFrame, optional): List of DataFrames to overlay on the plot.
    - labels (list of str, optional): Labels for the overlaid DataFrames (must match df_list length).
    - colors (str or list of str, optional): Color(s) for the overlaid DataFrames. If a string, it represents the
      column name in pareto_front from which to derive color values. If a list, it provides custom colors for
      each DataFrame in df_list.
    - color_group (list of str, optional): Default colors for each DataFrame in df_list if colors is not provided.
    - alpha_list (list of float, optional): Alpha (transparency) values for each overlaid DataFrame.
    - figsize (tuple, optional): Figure size (width, height) for the plot.
    - dot_size (int, optional): Size of the dots representing data points in the plot.

    Returns:
    - ax (matplotlib.axes._axes.Axes): Matplotlib Axes object.

    If df_list is provided, the function overlays multiple DataFrames on the same plot, using colors and labels
    specified. If colors is None and color_group is not provided, default colors are used.

    If colors is a string representing a column name in pareto_front, the function uses a colormap to color
    the Pareto front points based on the values in that column, and a color scale is added to the right side of
    the plot.

    Note: The function assumes that matplotlib (plt) has been imported.
    """
    if color_group is None:
        color_group = ["black", "black", "green", "blue"]

    if colors is None:
        colors = color_group
    if alpha_list is None:
        alpha_list = [0.05, 1, 0.05, 0.05]

    fig, ax = plt.subplots(figsize=figsize)  # Create a figure and axes object

    if df_list is not None:
        for idx, df in enumerate(df_list):
            if idx == 0:
                if colors is not None and isinstance(colors, str) and colors in pareto_front.columns:
                    color_values = df[colors].values
                    norm = plt.Normalize(color_values.min(), color_values.max())
                    cmap = plt.get_cmap('viridis')
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])

                    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
                    cbar.set_label(colors)

                    ax.scatter(df[x_column], df[y_column], c=color_values, s=dot_size,
                               cmap=cmap, label=labels[idx])
                else:
                    ax.scatter(df[x_column], df[y_column], c=color_group[idx], s=dot_size,
                               alpha=alpha_list[idx], label=labels[idx])
            else:
                ax.scatter(df[x_column], df[y_column], c=color_group[idx], s=dot_size,
                           alpha=alpha_list[idx], label=labels[idx])

    ax.plot(pareto_front[x_column], pareto_front[y_column], c='black', alpha=.1)
    ax.scatter(pareto_front[x_column], pareto_front[y_column], c='darkred', s=dot_size,
               label=front_label)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    return ax  # Return the Axes object


def pareto_front_3d(df, x_col, y_col, z_col=None, maximize_y=False):
    """
    Plots the Pareto front of a set of candidates based on their objective values.

    Args:
    - df: a pandas dataframe containing the objective values of the candidates.
    - x_col: the name of the column in candidates_df that contains the x-axis objective values.
    - y_col: the name of the column in candidates_df that contains the y-axis objective values.
    - z_col: (optional) the name of the column in candidates_df that contains the z-axis objective values.
    - maximize_y: a boolean indicating whether the y-axis objective should be maximized or minimized.

    Returns:
    - None
    """

    sorted_df = df.sort_values(by=[x_col, y_col], ascending=[True, maximize_y])
    x_ = np.array(sorted_df[x_col])
    y_ = np.array(sorted_df[y_col])
    z_ = None
    if z_col:
        z_ = np.array(sorted_df[z_col])

    pareto_front = np.array([[x_[0], y_[0]]])
    if z_col:
        pareto_front = np.array([[x_[0], y_[0], z_[0]]])

    for i in range(1, len(x_)):
        if maximize_y:
            if x_[i] >= pareto_front[-1][0] and y_[i] >= pareto_front[-1][1]:
                if z_col:
                    pareto_front = np.append(pareto_front, [[x_[i], y_[i], z_[i]]], axis=0)
                else:
                    pareto_front = np.append(pareto_front, [[x_[i], y_[i]]], axis=0)

        elif not maximize_y:
            if x_[i] >= pareto_front[-1][0] and y_[i] <= pareto_front[-1][1]:
                if z_col:
                    pareto_front = np.append(pareto_front, [[x_[i], y_[i], z_[i]]], axis=0)
                else:
                    pareto_front = np.append(pareto_front, [[x_[i], y_[i]]], axis=0)

    pareto_front = np.unique(pareto_front, axis=0)

    pareto_idx = []
    for idx in np.arange(len(pareto_front)):
        if z_col:
            loc_bool_df = df[[x_col, y_col, z_col]] == pareto_front[idx]
            pareto_idx.append(loc_bool_df[loc_bool_df[x_col] == True].index[0])

        else:
            loc_bool_df = df[[x_col, y_col]] == pareto_front[idx]
            pareto_idx.append(loc_bool_df[loc_bool_df[x_col] == True].index[0])

    index_values = np.array(pareto_idx)
    if z_col:
        return pd.DataFrame(pareto_front, columns=[x_col, y_col, z_col], index=index_values)
    else:
        return pd.DataFrame(pareto_front, columns=[x_col, y_col], index=index_values)


def plot_front_3d(
    pareto_front, df_list=None, loc_cols=None, front_label='Surface Front', labels=None, colors=None, alpha_list=None, figsize=(10, 10),
    dot_size=10
):
    if loc_cols == None:
        x_col = pareto_front.columns.values[0]
        y_col = pareto_front.columns.values[1]
        z_col = pareto_front.columns.values[2]

        front = pareto_front.values

    else:
        x_col, y_col, z_col = loc_cols

        front = pareto_front.loc[:, [x_col, y_col, z_col]].values

    if colors is None:
        colors = ["black", "black", "green", "blue"]
    if alpha_list is None:
        alpha_list = [0.05, 1, 0.05, 0.05]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if df_list is not None:
        for idx, df in enumerate(df_list):
            ax.scatter(df[x_col], df[y_col], df[z_col], c=colors[idx], alpha=alpha_list[idx], label=labels[idx])
    else:
        pass

    ax.scatter(front[:, 0], front[:, 1], front[:, 2], c="r", s=dot_size, label=front_label)
    #ax.plot(front[:, 0], front[:, 1], front[:, 2], c="r")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)

    if df_list is not None:
        # lim values
        lim_df = pd.concat(df_list)
        lim_df = lim_df[[x_col, y_col, z_col]].values

        ax.set_xlim3d(np.min(lim_df[:, 0]), np.max(lim_df[:, 0]))
        ax.set_ylim3d(np.min(lim_df[:, 1]), np.max(lim_df[:, 1]))
        ax.set_zlim3d(np.min(lim_df[:, 2]), np.max(lim_df[:, 2]))
    return ax


def plot_gitter_surface(pareto_front, df, edgecolor="black", alpha=0.5):
    # Create a Delaunay triangulation of the points

    x_col = pareto_front.columns.values[0]
    y_col = pareto_front.columns.values[1]
    z_col = pareto_front.columns.values[2]

    front = pareto_front.values
    points = df[[x_col, y_col, z_col]].values

    tri = Delaunay(points)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the points as a scatter plot

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=alpha)
    ax.scatter(front[:, 0], front[:, 1], front[:, 2], c="red", s=50)
    ax.plot(front[:, 0], front[:, 1], front[:, 2], c="red")

    # Plot the triangles as a gitter surface
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri.simplices, edgecolor=edgecolor, alpha=alpha)

    # Set the axis labels and limits
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_xlim3d(np.min(points[:, 0]), np.max(points[:, 0]))
    ax.set_ylim3d(np.min(points[:, 1]), np.max(points[:, 1]))
    ax.set_zlim3d(np.min(points[:, 2]), np.max(points[:, 2]))

    # Show the plot
    plt.show()


def return_opt_sample_pred_lists(args):

    model_types = args.eval.model_types
    opt_alg = args.opt.opt_type

    # use ModelWrapper Class
    wrapper = ModelWrapper(main_path=args.main_path,
                           file_name=args.file_name,
                           model_types=model_types,
                           model_sub_folder=None,
                           verbose=False)

    abs_output_dir = os.path.normpath(os.getcwd())

    if not os.path.isdir(abs_output_dir):
        print(f'Path does not exist: {abs_output_dir}')
        return [], []

    else:
        load_path = f'{abs_output_dir}/{args.file_name}/{args.splitfolder}/{opt_alg}'

        sample_list = []
        pred_list = []

        model_types_updated = model_types

        for i, model_type in enumerate(model_types):
            file_path = os.path.join(load_path, model_type, 'X_samples.h5')

            if not os.path.isfile(file_path):
                print(f'Solution candidates for {model_type} not available')
                continue

            loc_sample = pd.read_hdf(f'{file_path}')
            loc_pred = wrapper.return_predictions(loc_sample)[i]

            sample_list.append({"model_type": model_type,
                                "optimisation_strategy": opt_alg,
                                "samples": loc_sample
                                })
            pred_list.append({"model_type": model_type,
                              "optimisation_strategy": opt_alg,
                              "predictions": loc_pred
                              })

    return sample_list, pred_list