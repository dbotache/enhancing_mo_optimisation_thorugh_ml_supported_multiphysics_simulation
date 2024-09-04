
import pandas as pd
from pyparsing import Iterable


class PandasFileAdapter:
    file_type: str
    data_files: list

    def __init__(self, file_type: str, data_files: list) -> None:
        """Initialize the file adapter."""
        assert file_type in ["h5", "csv"]
        self.file_type = file_type
        self.data_files = data_files

    def load_data(self) -> Iterable:
        """Load the data from a file."""
        if self.file_type == "h5":
            dfs = [pd.read_hdf(data_file).infer_objects() for data_file in self.data_files]
        else:
            dfs = [pd.read_csv(data_file).infer_objects() for data_file in self.data_files]
        return pd.concat(dfs, axis=1)  # assume that we have to fuse dfs with different columns
