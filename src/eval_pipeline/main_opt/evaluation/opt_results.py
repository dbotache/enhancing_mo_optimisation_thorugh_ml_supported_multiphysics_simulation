import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


class GenerateParetoFront:
    """Print metrics and plot scores."""

    def __init__(self, pareto_vars: list, save_csv: str = None) -> None:
        """Initialize class."""
        self.pareto_vars = pareto_vars
        self.save_csv = save_csv

    def plot_3d(self, df_optimized: pd.DataFrame, df_gt: pd.DataFrame) -> None:
        # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print(df_optimized[self.pareto_vars[2]])

        alpha = df_optimized["marker"].map({"red": 1.0, "black": 0.5, "#DDDDDD": 0.35})
        s = df_optimized["marker"].map({"red": 30.0, "black": 5.0, "#DDDDDD": 5.0})

        ax.scatter(df_optimized[self.pareto_vars[0]], df_optimized[self.pareto_vars[1]], df_optimized[self.pareto_vars[2]], c=df_optimized["marker"], alpha=alpha, s=s)
        ax.set_xlabel(self.pareto_vars[0])
        ax.set_ylabel(self.pareto_vars[1])
        ax.set_zlabel(self.pareto_vars[2])
        return fig

    def plot_2d(self, df_optimized: pd.DataFrame, df_gt: pd.DataFrame) -> None:
        fig = plt.figure()
        alpha = df_optimized["marker"].map({"red": 1.0, "black": 0.5, "#DDDDDD": 0.35})

        plt.scatter(df_optimized[self.pareto_vars[0]], df_optimized[self.pareto_vars[1]], c=df_optimized["marker"], alpha=alpha)
        plt.xlabel(self.pareto_vars[0])
        plt.ylabel(self.pareto_vars[1])
        return fig

    def __call__(self, df_optimized: pd.DataFrame,
                 df_gt: pd.DataFrame) -> None:
        """Log metrics to wandb."""
        assert len(self.pareto_vars) in [2, 3]

        # TODO: fix this in opt script
        df_optimized["satisfied_constraints"] = df_optimized["satisfied_constraints"].map({"tensor(True)": True, "tensor(False)": False, "True": True, "False": False})
        
        # mark out the points that don't satisfy the constraints
        df_optimized["marker"] = df_optimized["satisfied_constraints"].map({True: "black", False: "#DDDDDD"})
        df_optimized["pareto_optimal"] = False

        for i in range(len(df_optimized)):
            if not df_optimized["satisfied_constraints"].iloc[i]:
                # exclue points that don't satisfy the constraints
                continue

            pareto_criteria = np.delete(np.array(df_optimized["satisfied_constraints"]), i)

            for p_var in self.pareto_vars:
                others_pvar = np.delete(np.array(df_optimized[p_var]), i)
                self_pvar = df_optimized[p_var].iloc[i]
                pareto_criteria = np.logical_and(pareto_criteria, others_pvar < self_pvar)

            if np.any(pareto_criteria):
                continue

            # found pareto point
            df_optimized["marker"].iloc[i] = "red"
            df_optimized["pareto_optimal"].iloc[i] = True

        if len(self.pareto_vars) == 2:
            fig = self.plot_2d(df_optimized, df_gt)
            wandb.log({"pareto plot": wandb.Image(fig)})
            plt.close(fig)

        elif len(self.pareto_vars) == 3:
            fig = self.plot_3d(df_optimized, df_gt)
            wandb.log({"pareto plot": wandb.Image(fig)})
            plt.close(fig)
        
        if self.save_csv is not None:
            df_optimized.to_csv(self.save_csv, index=False)


class GenerateViolinPlots:
    """Print metrics and plot scores."""
    def violin_plot(self, col_gt, col_optimized, col):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.violinplot(data=col_gt.values, ax=ax)
        sns.violinplot(data=col_optimized.values, ax=ax, color='red')
        plt.title(col)
        return fig
    
    def __call__(self, df_optimized: pd.DataFrame,
                 df_gt: pd.DataFrame) -> None:
        """Log metrics to wandb."""
        for col in df_optimized.columns:
            if col in df_gt.columns:
                fig = self.violin_plot(df_gt[col], df_optimized[col], col)
                wandb.log({"violin_plot": wandb.Image(fig)})
                plt.close(fig)


class OptResultsStep:
    """Step to evaluate a model against a file."""

    optimized_file_adapter: any
    gt_file_adapters: list[any]
    vis_hooks: list[callable]

    def __init__(self, optimized_file_adapter: any, 
                 gt_file_adapters: list[any], 
                 vis_hooks: list[callable]) -> None:
        """Initialize the step."""
        self.optimized_file_adapter = optimized_file_adapter
        self.gt_file_adapters = gt_file_adapters
        self.vis_hooks = vis_hooks

    def run(self) -> None:
        """Run the step."""
        dfs_gt = []
        for file_adapter in self.gt_file_adapters:
            dfs_gt.append(file_adapter.load_data())

        df_gt = pd.concat(dfs_gt, axis=0)
        df_optimized = self.optimized_file_adapter.load_data()

        for vis_hook in self.vis_hooks:
            vis_hook(df_optimized.copy(), df_gt.copy())
