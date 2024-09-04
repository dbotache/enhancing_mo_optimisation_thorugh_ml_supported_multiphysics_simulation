import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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


class ParameterData(Dataset):
    def __init__(self, X, y, n=1):

        """
        A custom PyTorch Dataset that formats input and target data as (X, y) pairs.

        Args:

        X (pandas DataFrame): input data
        y (pandas DataFrame): target data
        n (int, optional): number of time steps to include in each sample (default=1)
        Returns:

        X_sample, y_sample (tuple): tuple containing the input and target data for a single sample
        Note:

        The input data is converted to a numpy array using the values attribute of pandas DataFrame.
        The target data is also converted to a numpy array using the values attribute of pandas DataFrame.
        The index of the target DataFrame is stored in self.index.
        The length of the dataset is determined by the number of rows in the target DataFrame.
        The getitem method returns a tuple of (X_sample, y_sample) where:
        X_sample is a numpy array of shape (n, num_features) where n is the number of time steps
        and num_features is the number of features in the input data.
        y_sample is a numpy array of shape (n, num_targets) where n is the number of time steps
        and num_targets is the number of targets in the target data.
        The X_sample and y_sample arrays are slices of the original input and target arrays starting
        at index 'idx' and ending at index 'idx + n'.
        This class is used in conjunction with PyTorch DataLoader to efficiently load data for model training.
        """

        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.DataFrame): y = y.values

        self.X = X
        self.y = y
        self.n_steps = n

    def __len__(self):
        if self.n_steps == 1:
            return len(self.y)
        else:
            return len(self.y) - self.n_steps

    def __getitem__(self, idx):
        return self.X[idx:idx + self.n_steps], self.y[idx:idx + self.n_steps]
