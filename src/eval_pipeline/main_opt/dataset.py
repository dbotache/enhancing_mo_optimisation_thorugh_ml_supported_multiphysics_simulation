from __future__ import annotations


import numpy as np
import pandas as pd
import torch


class PandasDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, input_cols: list,
                 target_cols: list) -> None:
        """Custom dataset for pandas dataframes."""
        self.df_input = df[input_cols].astype(object)
        self.df_input = [
            {k: torch.tensor(self.df_input.iloc[idx][k])
             for k in self.df_input.columns}
            for idx in range(len(self.df_input))
        ]
        self.y = torch.tensor(df[target_cols].copy()
                              .to_numpy().astype(np.float32))

    def __len__(self) -> int:
        """Returns the number of data points in the dataset."""
        return len(self.df_input)

    def __getitem__(self, idx: int) -> tuple[np.array, np.array]:
        """Returns the features, the targets and weighting of the targets."""
        return self.df_input[idx], self.y[idx]
