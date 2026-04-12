import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

from typing import Union, Dict
from omegaconf import DictConfig

VALID_SUBDATASETS = {"h1", "h2", "m1", "m2"}


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
                 normalization_params: Union[Dict, DictConfig]=None):
        
        super(ETTDataset, self).__init__()

        # check if subdataset is valid
        if subdataset not in VALID_SUBDATASETS:
            raise ValueError(f"Invalid subdataset {subdataset}. Must be one of {VALID_SUBDATASETS}")
        


        path_to_data = Path(f"data/ETT/ETT{subdataset}.csv")

        if not path_to_data.exists():
            raise FileNotFoundError(f"Data file {path_to_data} not found. Please make sure the ETT dataset is downloaded and placed in the correct directory.")

        self.data = pd.read_csv(path_to_data)
        self.timestamp = self._form_timestamp()
        self.target = self.data["OT"]

        self.data = self.data.drop(columns=["date", "OT"])
        self._normalize_data(normalization_type, normalization_params)

        # make torch tensors
        self.timestamp = torch.tensor(self.timestamp, dtype=torch.float32)
        self.data = torch.tensor(self.data.values, dtype=torch.float32)
        self.target = torch.tensor(self.target.values, dtype=torch.float32)

    def _form_timestamp(self):
        """Convert the 'date' column to integer timestamps (seconds)."""
        # Convert to pandas datetime, then to integer nanoseconds and to seconds
        timestamps = pd.to_datetime(self.data["date"])
        # Convert to nanoseconds (int64) and then to seconds
        timestamp = (timestamps.astype('int64') // 10**9).to_numpy()
        return timestamp

    def _normalize_data(self, normalization_type: str, normalization_params: Union[Dict, DictConfig]) -> None:
        
        # check if normalization type is supported
        if normalization_type not in ["minmax", "standard", "none"]:
            raise NotImplementedError(f"The normalization type {normalization_type} is not implemented yet")
        
        # check if normalization params are provided, if so use them, otherwise compute from data
        
        # the logics behind this block is that we want to be able to use the same normalization params for train/val/test splits, which are computed from the train split, and passed to the val/test splits via the config
        if normalization_params is not None:
            if normalization_type == "minmax":
                self.data = (self.data - normalization_params["data_min"]) / (normalization_params["data_max"] - normalization_params["data_min"])
                self.target = (self.target - normalization_params["target_min"]) / (normalization_params["target_max"] - normalization_params["target_min"])
                self.timestamp = (self.timestamp - normalization_params["timestamp_min"]) / (normalization_params["timestamp_max"] - normalization_params["timestamp_min"])
                return 
            elif normalization_type == "standard":
                self.data = (self.data - normalization_params["data_mean"]) / normalization_params["data_std"]
                self.target = (self.target - normalization_params["target_mean"]) / normalization_params["target_std"]
                self.timestamp = (self.timestamp - normalization_params["timestamp_mean"]) / normalization_params["timestamp_std"]
                return
        else:

            if normalization_type == "minmax":
                self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
                self.target = (self.target - self.target.min()) / (self.target.max() - self.target.min())
                self.timestamp = (self.timestamp - self.timestamp.min()) / (self.timestamp.max() - self.timestamp.min())
                return
            elif normalization_type == "standard":
                self.data = (self.data - self.data.mean()) / self.data.std()
                self.target = (self.target - self.target.mean()) / self.target.std()
                self.timestamp = (self.timestamp - self.timestamp.mean()) / self.timestamp.std()
                return
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.timestamp[idx], self.data[idx], self.target[idx]


if __name__ == "__main__":
    subdataset = "h1"
    print("Testing ETTDataset with subdataset:", subdataset)
    dataset = ETTDataset(subdataset="h1")
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])

    for elem in dataset:
        for subelem in elem:
            assert ~(torch.isnan(subelem)).any(), f"NaN found in ETT{subdataset} dataset"
