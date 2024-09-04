import ipywidgets as widgets
import matplotlib.pyplot as plt


def interactive_plot(X, y, scatter=True, sort=True, sort_targets=False, figsize=(15, 5)):
    @widgets.interact(feature_name=widgets.Dropdown(
        options=X.columns.values,
        value=X.columns.values[0],
        description='Parameter:',
        disabled=False,
    ),
        target_name=widgets.Dropdown(
            options=y.columns.values,
            value=y.columns.values[0],
            description='Target:',
            disabled=False,
        ))
    def plot_(feature_name, target_name):
        loc_df = X.loc[:, [feature_name]]
        loc_df[target_name] = y.loc[:, [target_name]]
        
        f, ax = plt.subplots(figsize=figsize)

        if sort:
            if sort_targets:
                loc_df = loc_df.sort_values(by=[target_name], ascending=True)
                
                if scatter:
                    ax.scatter(loc_df.loc[:, [target_name]], loc_df.loc[:, [feature_name]])
                else:
                    ax.plot(loc_df.loc[:, [target_name]], loc_df.loc[:, [feature_name]])
                    
                ax.set_ylabel(feature_name)
                ax.set_xlabel(target_name)
            else:
                loc_df = loc_df.sort_values(by=[feature_name], ascending=True)
                if scatter:
                    ax.scatter(loc_df.loc[:, [feature_name]], loc_df.loc[:, [target_name]])
                else:
                    ax.plot(loc_df.loc[:, [feature_name]], loc_df.loc[:, [target_name]])

                ax.set_xlabel(feature_name)
                ax.set_ylabel(target_name)
