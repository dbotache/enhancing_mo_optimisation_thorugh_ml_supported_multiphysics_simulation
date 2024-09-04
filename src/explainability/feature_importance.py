import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_feature_importance(X, y, xgb_object):

    importance_list = []
    for reg_ in xgb_object.model_container:
        importance_list.append(reg_.feature_importances_)

    importance_matrix = np.vstack(importance_list)

    x_cols = X.columns
    y_cols = y.columns

    name_legend = "Objective-Target"

    loc_importance = pd.DataFrame(importance_matrix, columns=x_cols, index=y_cols)
    feature_importance_df = loc_importance.reset_index().rename(columns={"index": name_legend})

    return feature_importance_df


def plot_feature_importance(df, name_legend="Objective-Target", type='bars', save_path='/temp'):

    if type == 'heatmap':

        importance_df = df

        plt.style.use('seaborn')
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica",
            "axes.labelsize": 10,
            'axes.titlesize': 12,
            'axes.xmargin': 0.05,
            'axes.ymargin': 0.01,
            'axes.zmargin': 0.01,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 7.5,
        })

        plt.figure()
        sns.heatmap(importance_df.set_index('Objective-Target'))
        plt.tight_layout()

        plt.savefig(f'{save_path}/hm_bars_feature_relevances.svg')
        plt.savefig(f'{save_path}/hm_bars_feature_relevances.pdf')

        plt.show()

    elif type == 'bars':

        df_list = []
        for col in df.drop(columns=name_legend).columns.values:
            loc_df = df.loc[:, [col, name_legend]]
            loc_df = loc_df.rename(columns={col: "Feature Relevances"})
            loc_df['Input Feature'] = col
            df_list.append(loc_df)

        importance_df = pd.concat(df_list)

        plt.style.use('seaborn')
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica",
            "axes.labelsize": 14,
            'axes.titlesize': 16,
            'axes.xmargin': 0.05,
            'axes.ymargin': 0.01,
            'axes.zmargin': 0.01,
        })
        # create the barplot
        plt.figure(figsize=(7, 3.5))

        # create the stacked barplot
        ax = sns.barplot(data=importance_df, x='Input Feature', y="Feature Relevances", hue=name_legend, dodge=False)

        # rotate the x labels by 90 degrees
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        plt.legend(ncol=5)
        plt.tight_layout()

        plt.savefig(f'{save_path}/bars_feature_relevances.svg')
        plt.savefig(f'{save_path}/bars_feature_relevances.pdf')

        # display the plot
        plt.show()
