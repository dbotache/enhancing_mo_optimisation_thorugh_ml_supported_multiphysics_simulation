from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pandas import DataFrame

epsilon_constraints = 1e-4


@dataclass
class OptResult:
    dict_in: dict
    dict_out: dict
    objectives: list
    other_attributes: dict


def load_checkpoint_from_artifact(artifact_key: str) -> Path:
    """Load artifact."""
    api = wandb.Api()
    artifact = api.artifact(artifact_key, type="model")
    artifact_dir = artifact.download()
    return Path(artifact_dir) / "last.ckpt"


def data_to_series(result: OptResult) -> DataFrame:
    """Convert OptResult to pandas Series."""
    return (
        pd.Series(
            {
                **result.dict_in,
                **result.dict_out,
                **result.other_attributes,
                # "pareto_optimal": result.pareto_optimal,
                "satisfied_constraints": result.objectives[0]
                < epsilon_constraints,  # assume that the first objective is the constraint
            }
        )
        .to_frame()
        .T
    )


def save_results(results: dict, save_key: str) -> None:
    """Save results to csv."""
    series = [data_to_series(res) for res in results]
    df_result = pd.concat(series, axis=0)
    df_result.to_csv(f"{save_key}.csv")
    df_result.T.to_csv(f"{save_key}T.csv")


def get_transformed_targets(encoder) -> tuple:
    """Get transformed targets."""
    c1 = encoder.encoder["M(S2,n1)"]["scaler"].transform(np.array([323.0]).reshape(-1, 1))
    c2 = encoder.encoder["M(S2,n2)"]["scaler"].transform(np.array([321.0]).reshape(-1, 1))
    c3 = encoder.encoder["M(S2,n3)"]["scaler"].transform(np.array([191.0]).reshape(-1, 1))
    c4 = encoder.encoder["M(S2,n4)"]["scaler"].transform(np.array([94.0]).reshape(-1, 1))

    return c1, c2, c3, c4


def get_objectives(litmlp: pl.LightningModule) -> list[callable]:
    """Get objectives."""
    c1, c2, c3, c4 = get_transformed_targets(litmlp.encoder)

    def o_1(dict_out_norm):
        ref_zero = torch.zeros_like(dict_out_norm["M(S2,n1)"])

        cdif1 = torch.max(c1.item() - dict_out_norm["M(S2,n1)"], ref_zero)
        cdif2 = torch.max(c2.item() - dict_out_norm["M(S2,n2)"], ref_zero)
        cdif3 = torch.max(c3.item() - dict_out_norm["M(S2,n3)"], ref_zero)
        cdif4 = torch.max(c4.item() - dict_out_norm["M(S2,n4)"], ref_zero)

        return cdif1 + cdif2 + cdif3 + cdif4

    def o_2(dict_out_norm):
        return (dict_out_norm["Pv_Antrieb_Fzg_Zykl_1"] + dict_out_norm["Pv_Antrieb_Fzg_Zykl_2"]) / 2.0

    def o_3(dict_out_norm):
        return dict_out_norm["MEK_Aktivteile"]

    return [o_1, o_2, o_3]


def evaluate_objectives(litmlp: pl.LightningModule, objectives: list[callable], dict_in: dict) -> tuple:
    """Get objective and constraint values."""
    y_hat = litmlp(dict_in)
    dict_out_norm = dict(zip(litmlp.encoder.target_cols, y_hat.squeeze()))
    eval_objectives = torch.stack([o(dict_out_norm) for o in objectives])

    y_hat_org = litmlp.inverse_transform_targets(y_hat.clone().detach().cpu())
    dict_out = dict(zip(litmlp.encoder.target_cols, y_hat_org.squeeze()))

    return (
        eval_objectives,
        {k: v.item() for k, v in dict_out.items()},
    )
