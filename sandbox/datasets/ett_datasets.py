import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

from typing import Union, DictConfig

class ETTDataset(Dataset):
    """
    PyTorch Dataset for the Electricity Transformer Temperature (ETT) dataset.

    The ETT dataset contains time series data of electricity transformer temperature
    and other load features, collected from two separate counties in China.
    It includes four sub-datasets: ETTh1, ETTh2 (hourly), ETTm1, ETTm2 (15-minute).

    Source: https://github.com/zhouhaoyi/ETDataset
    The dataset files are expected to be located in `data/ETT/` directory.

    Parameters
    ----------
    subdataset : str, optional
        Which sub-dataset to load. Must be one of {"h1", "h2", "m1", "m2"}.
        Default is "h1".
    normalization_type : str, optional
        Type of normalization to apply. Currently only "minmax" is supported.
        Default is "minmax".

    Attributes
    ----------
    data : pd.DataFrame
        Normalized feature columns.
    target : pd.Series
        Normalized target column "OT" (Oil Temperature).
    timestamp : np.ndarray
        Normalized integer timestamps derived from the "date" column.

    Examples
    --------
    >>> dataset = ETTDataset(subdataset="h1")
    >>> len(dataset)
    17420
    >>> timestamp, features, target = dataset[0]
    """

    def __init__(self, subdataset: str = "h1", 
                 normalization_type: str = "none",
                 normalization_params: Union[dict, DictConfig]=None):
        super(ETTDataset, self).__init__()
        path_to_data = Path(f"data/ETT/ETT{subdataset}.csv")
        self.data = pd.read_csv(path_to_data)
        self._form_timestamp()
        self.data = self.data.drop(columns=["date"])
        self.normalization_params = self._normalize_data(normalization_type)
        
        # make torch tensors
        self.timestamp = torch.tensor(self.timestamp, dtype=torch.float32)
        self.data = torch.tensor(self.data.values, dtype=torch.float32)
        self.target = torch.tensor(self.target.values, dtype=torch.float32)
    def _form_timestamp(self):
        """Convert the 'date' column to integer timestamps (seconds)."""
        # Convert to pandas datetime, then to integer nanoseconds and to seconds
        timestamps = pd.to_datetime(self.data["date"])
        # Convert to nanoseconds (int64) and then to seconds
        self.timestamp = (timestamps.astype('int64') // 10**9).to_numpy()

    def _normalize_data(self, normalization_type: str):
        if normalization_type not in ["minmax", "none"]:
            raise NotImplementedError("The normalization type is not implemented yet")
        if normalization_type == "minmax":
            self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
            self.target = (self.target - self.target.min()) / (self.target.max() - self.target.min())
            self.timestamp = (self.timestamp - self.timestamp.min()) / (self.timestamp.max() - self.timestamp.min())
            self.norma
        elif normalization_type == "none":
            return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.timestamp[idx], self.data[idx], self.target[idx]


if __name__ == "__main__":
    subdataset = "h1"
    dataset = ETTDataset(subdataset="h1")
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    for elem in dataset:
        for subelem in elem:
            assert ~(torch.isnan(subelem)).any(), f"NaN found in ETT{subdataset} dataset"
