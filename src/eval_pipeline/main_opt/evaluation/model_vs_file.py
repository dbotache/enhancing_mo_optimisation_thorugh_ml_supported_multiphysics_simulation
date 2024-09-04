from math import sqrt
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from pytorch_lightning import LightningModule
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score



class PrintMetricsPlotScores:
    """Print metrics and plot scores."""

    def __init__(self, scores: list) -> None:
        """Initialize class."""
        self.scores = scores

    # ies function to evaluate regression models
    def evaluate_reg(self, y_true: pd.DataFrame, y_pred: pd.DataFrame, column: str) -> pd.DataFrame:
        """Evaluate regression models."""
        # cols need to be numeric
        y_pred = np.array(y_pred[column].astype(float))
        y_true = np.array(y_true[column].astype(float))

        bias = (y_pred - y_true).sum() * (1 / len(y_true))
        rmse = MSE(y_true, y_pred) ** (1 / 2)
        mape = MAPE(y_true, y_pred)
        nbias = bias / y_true.mean()
        mae = MAE(y_true, y_pred)
        nmae = mae / y_true.mean()
        mse = MSE(y_true, y_pred)
        nmse = mse / y_true.mean()
        nrmse = rmse / y_true.mean()
        r2 = r2_score(y_true, y_pred)
        max_deviation = np.abs(y_pred - y_true).max() / y_true.mean()
        min_deviation = np.abs(y_pred - y_true).min() / y_true.mean()

        return pd.DataFrame(
            {
                "target": column,
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
                "max_deviation(%)": max_deviation,
                "min_deviation(%)": min_deviation,
            },
            index=[column],
        )

    def __call__(
        self,
        df: pd.DataFrame,
        df_pred: pd.DataFrame,
    ) -> None:
        """Log metrics to wandb."""

        scores_list = []
        for col in df_pred.columns:
            if col in df.columns:
                scores_list.append(self.evaluate_reg(df, df_pred, col))

        scores_df = pd.concat(scores_list)
        table = wandb.Table(dataframe=scores_df)
        wandb.log({"metrics": table})

        for score in self.scores:
            f, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f"Score: {score}")
            ax.set_xticklabels(scores_df.index, rotation=90)
            sns.barplot(x="target", y=score, data=scores_df)
            ax.set_xlabel("target")
            f.tight_layout()
            wandb.log({f"{score}_plot": wandb.Image(f)})


class GTvsPredPlot:
    """Print metrics and plot scores."""

    def satisfies_constraints(self, df: pd.DataFrame) -> np.array:
        """Returns the constraint."""
        ref_zero = np.zeros_like(df["M(S2,n1)"])
        c1 = np.maximum(323.0 - df["M(S2,n1)"].astype(np.float), ref_zero)
        c2 = np.maximum(321.0 - df["M(S2,n2)"].astype(np.float), ref_zero)
        c3 = np.maximum(191.0 - df["M(S2,n3)"].astype(np.float), ref_zero)
        c4 = np.maximum(94.0 - df["M(S2,n4)"].astype(np.float), ref_zero)

        return c1 + c2 + c3 + c4

    def __call__(
        self,
        df: pd.DataFrame,
        df_pred: pd.DataFrame,
    ) -> None:
        """Log metrics to wandb."""
        common_cols = set(df.columns).intersection(set(df_pred.columns))

        num_axes = len(common_cols)
        num_axes_x = int(sqrt(num_axes))
        num_axes_y = int(sqrt(num_axes))
        if num_axes_x * num_axes_y < num_axes:
            num_axes_y += 1
        if num_axes_x * num_axes_y < num_axes:
            num_axes_x += 1

        df["satisfies_constraints"] = (self.satisfies_constraints(df) <= 0.001).astype(int)

        x_line = {"M(S2,n1)": 323.0, "M(S2,n2)": 321.0, "M(S2,n3)": 191.0, "M(S2,n4)": 94.0}
        f, axes = plt.subplots(num_axes_x, num_axes_y, figsize=(4 * num_axes_y + 1, 4 * num_axes_x))
        for key, ax in zip(common_cols, axes.flatten()):
            df[key] = df[key].astype(float)
            df_pred[key] = df_pred[key].astype(float)

            if key in x_line:
                ax.axvline(x=x_line[key], color="black", label="constraint")
                ax.axhline(y=x_line[key], color="black", label="constraint")

            ax.set_xlabel(f"{key} - value from file")
            ax.set_ylabel(f"{key} - predicted value")

            ax.scatter(
                df[key],
                df_pred[key],
                alpha=0.65,
            )
            ax.scatter(
                df[key],
                df_pred[key],
                facecolors="none",
                edgecolors=["red" if x else "none" for x in df["satisfies_constraints"]],
                linewidth=1.5,
            )
            tmp = np.linspace(min(df[key]), max(df[key]), num=100)
            ax.plot(tmp, tmp, ls="--")

        f.suptitle("Model vs. file.")
        f.tight_layout()

        wandb.log({"prediction_vs_gt": wandb.Image(f)})


class TorchScriptInference:
    def load_model(self, model_file: str) -> LightningModule:
        """Load the model."""
        self.model = torch.jit.load(model_file)

    def load_data(self, df: pd.DataFrame) -> None:
        """Load the data from a pandas dataframe."""
        self.df = df

    def get_predictions(self) -> pd.DataFrame:
        """Get predictions for the test set."""
        litmodel = self.model

        inputs = []
        for col in litmodel.encoder.input_cols:
            inputs.append(torch.tensor(np.array(self.df[col]).astype(np.float32)))

        with torch.no_grad():
            y_hat = litmodel(torch.stack(inputs, axis=1))

        y_hat_result = {}
        for i, key in enumerate(litmodel.encoder.target_cols):
            y_hat_result[key] = np.array(y_hat[:, i])

        return pd.DataFrame(y_hat_result)


class EnsembleInference:
    def load_model(self, model_file: str) -> None:
        """Load the model."""
        # iterate over all subfolders in model_file
        model_file = Path(model_file)
        self.reg_list = {}
        for folder in model_file.glob("*"):
            if folder.is_dir():
                target_name = folder.name[len(model_file.name) + 1 :]

                with open(folder / "meta_ensemble.pkl", "rb") as f:
                    try:
                        reg_ = pickle.load(f)
                        self.reg_list[target_name] = reg_
                    except Exception as e:
                        print(f"Failed to load regressor for model {target_name} with exception", e)

    def load_data(self, df: pd.DataFrame) -> None:
        """Load the data from a pandas dataframe."""
        self.df = df

    def get_predictions(self) -> pd.DataFrame:
        """Get predictions for the test set."""
        y_hat_org_result = defaultdict(list)

        for target_name, reg_ in self.reg_list.items():
            y_hat_org_result[target_name] = reg_.predict(
                self.df[[col for col in self.df.columns if str(col).startswith("vdp")]]
            )

        return pd.DataFrame(y_hat_org_result)


class PandasFileAdapter:
    def __init__(self, file_type: str) -> None:
        """Initialize the file adapter."""
        assert file_type in ["h5", "csv"]
        self.file_type = file_type

    def load_data(self, data_file: str) -> Iterable:
        """Load the data from a file."""
        if self.file_type == "h5":
            return pd.read_hdf(data_file).infer_objects()
        else:
            return pd.read_csv(data_file).infer_objects()


class ModelVsFileStep:
    """Step to evaluate a model against a file."""

    model_file: str
    data_file: str
    model_adapter: any
    file_adapter: any
    vis_hooks: list[callable]

    def __init__(
        self, model_file: str, data_file: str, model_adapter: any, file_adapter: any, vis_hooks: list[callable]
    ) -> None:
        """Initialize the step."""
        self.model_file = model_file
        self.data_file = data_file
        self.model_adapter = model_adapter
        self.file_adapter = file_adapter
        self.vis_hooks = vis_hooks

    def run(self) -> None:
        """Run the step."""
        df = self.file_adapter.load_data(self.data_file)

        self.model_adapter.load_model(self.model_file)
        self.model_adapter.load_data(df)

        df_pred = self.model_adapter.get_predictions()

        for vis_hook in self.vis_hooks:
            vis_hook(df, df_pred)
