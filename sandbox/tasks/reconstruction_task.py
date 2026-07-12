"""Reconstruction task, preprocessing, and sliding-window dataset helpers.

This module is responsible for the task-specific part of the pipeline:

- reading the raw row-wise dataset configured under ``cfg.data``
- splitting the time series chronologically into train/val/test
- fitting normalization statistics on the train split only
- converting each split into sliding windows
- exposing loss/metrics utilities expected by ``BaseRunner`` or its ancestors

The raw dataset itself is intentionally simple and only provides
``(timestamp, features, target)`` rows. All reconstruction semantics live here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sandbox.tasks.forecasting_task import ForecastingTask

from ..utils.dataset_normalization import DatasetExtractor

TensorLike = Union[np.ndarray, torch.Tensor]


def _to_float_tensor(value: TensorLike) -> torch.Tensor:
    """Convert NumPy arrays or tensors to detached float32 CPU tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=torch.float32, device="cpu")
    return torch.as_tensor(value, dtype=torch.float32)


class LocalDataset(Dataset):
    """A lightweight in-memory dataset for already prepared tensors."""

    def __init__(
        self,
        timestamps: TensorLike,
        features: TensorLike,
    ) -> None:
        super().__init__()

        self.timestamps = _to_float_tensor(timestamps)
        self.features = _to_float_tensor(features)

        if self.timestamps.ndim != 1:
            raise ValueError("timestamps must be a 1D tensor")
        if self.features.ndim != 2:
            raise ValueError("features must be a 2D tensor with shape (N, F)")
        if not (len(self.timestamps) == len(self.features)):
            raise ValueError(
                "timestamps, features, and target must have the same length"
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.timestamps[idx], self.features[idx]


class BasicSlidingWindowDataset(Dataset):
    """Convert a row-wise dataset into reconstruction windows.

    Each sample is returned as a dictionary compatible with the repository
    batch contract:

    - ``x``: input context of shape ``(T, F)``
    - ``x_t``: input timestamps of shape ``(T,)``
    """

    def __init__(
        self,
        dataset: Dataset,
        stride: int,
        context_length: int,
        start_idx: int = 0,
        end_idx: int | None = None,
        part: str = "train",
        device: str = "cpu",
    ) -> None:
        super().__init__()

        if part not in {"train", "val", "test"}:
            raise ValueError("part must be one of 'train', 'val', or 'test'")
        self.part = part

        self.stride = stride
        self.context_length = context_length
        self.data = dataset
        self.start_idx = start_idx
        self.end_idx = len(dataset) if end_idx is None else end_idx

        self._configure_windows()
        self.timestamps, self.inputs = self._collect_data()
        self.timestamps = self.timestamps.to(device)
        self.inputs = self.inputs.to(device)

    def _configure_windows(self) -> None:
        last_start = self.end_idx - self.context_length
        if last_start <= self.start_idx:
            raise ValueError(
                "Split is too short to build even one sliding window with the "
                "requested context_length and horizon."
            )

        self.start_indices = torch.arange(
            self.start_idx, last_start, self.stride, dtype=torch.long
        )
        self.end_indices = self.start_indices + self.context_length

    def _collect_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset_length = len(self.data)
        if dataset_length == 0:
            raise ValueError(
                "Sliding window dataset cannot be built from an empty dataset."
            )

        timestamps = []
        inputs = []

        for idx in range(dataset_length):
            timestamp, input_tensor = self.data[idx]
            timestamps.append(_to_float_tensor(timestamp))
            inputs.append(_to_float_tensor(input_tensor))

        return torch.stack(timestamps), torch.stack(inputs)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, dict]]:
        start_idx = int(self.start_indices[idx].item())
        end_idx = int(self.end_indices[idx].item())

        input_timestamp_window = self.timestamps[start_idx:end_idx]
        input_window = self.inputs[start_idx:end_idx]

        return {
            "x": input_window,
            "x_t": input_timestamp_window,
            "meta": {"split_part": self.part},
        }


class ReconstructionTask(ForecastingTask):
    """Task object that prepares reconstruction data and computes objectives."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__(config)

    def _instantiate_metrics(self):
        metric_names = self._metric_names()
        metric_fns = {}

        for metric_key, metric_name in metric_names.items():
            if metric_name == "mse":
                metric_fns[metric_key] = lambda prediction, batch: (
                    nn.functional.mse_loss(prediction, batch["x"])
                )
            elif metric_name == "mae":
                metric_fns[metric_key] = lambda prediction, batch: (
                    nn.functional.l1_loss(prediction, batch["x"])
                )
            else:
                raise NotImplementedError(f"Unsupported metric: {metric_name}")
        return metric_fns

    def loss_fn(self, outputs: torch.Tensor, batch: dict):
        """Compute task loss from model outputs and batch targets."""
        return self.loss(outputs, batch["x"])

    def _fit_normalization_stats(
        self,
        train_timestamps: torch.Tensor,
        train_features: torch.Tensor,
        normalization_type: str,
        normalize_timestamps: bool,
    ) -> Dict[str, torch.Tensor]:
        if normalization_type == "none":
            return {}

        eps = torch.tensor(1e-8, dtype=torch.float32)
        stats: Dict[str, torch.Tensor] = {}

        if normalization_type == "standard":
            stats["features_center"] = train_features.mean(dim=0)
            stats["features_scale"] = train_features.std(
                dim=0, unbiased=False
            ).clamp_min(eps)
            if normalize_timestamps:
                stats["timestamps_center"] = train_timestamps.mean()
                stats["timestamps_scale"] = train_timestamps.std(
                    unbiased=False
                ).clamp_min(eps)
        else:
            stats["features_center"] = train_features.min(dim=0).values
            stats["features_scale"] = (
                train_features.max(dim=0).values - train_features.min(dim=0).values
            ).clamp_min(eps)
            if normalize_timestamps:
                stats["timestamps_center"] = train_timestamps.min()
                stats["timestamps_scale"] = (
                    train_timestamps.max() - train_timestamps.min()
                ).clamp_min(eps)

        return stats

    def _apply_normalization(
        self,
        timestamps: torch.Tensor,
        features: torch.Tensor,
        normalization_type: str,
        stats: Mapping[str, torch.Tensor],
        normalize_timestamps: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if normalization_type == "none":
            return timestamps, features

        if normalization_type in {"standard", "minmax"}:
            norm_features = (features - stats["features_center"]) / stats[
                "features_scale"
            ]
            norm_timestamps = timestamps

            if normalize_timestamps:
                norm_timestamps = (timestamps - stats["timestamps_center"]) / stats[
                    "timestamps_scale"
                ]
        else:
            raise NotImplementedError(
                f"Unsupported normalization type: {normalization_type}"
            )
        return norm_timestamps, norm_features

    def build_dataloaders(self, cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Build chronological train/val/test dataloaders for reconstruction."""
        data_cfg = cfg["data"]
        dataset = instantiate(data_cfg["dataset"])

        timestamps, features, _ = DatasetExtractor(dataset).extract_data()
        timestamps = _to_float_tensor(timestamps)
        features = _to_float_tensor(features)

        train_slice, val_slice, test_slice = self._split_indices(len(dataset))

        train_timestamps = timestamps[train_slice]
        train_features = features[train_slice]

        val_timestamps = timestamps[val_slice]
        val_features = features[val_slice]

        test_timestamps = timestamps[test_slice]
        test_features = features[test_slice]

        normalization_cfg = self.config.get("normalization", {})
        if not isinstance(normalization_cfg, Mapping):
            normalization_cfg = {}

        normalization_type = self._instantiate_scaler(
            str(normalization_cfg.get("type", "none"))
        )
        normalize_timestamps = bool(normalization_cfg.get("timestamps", True))

        norm_stats = self._fit_normalization_stats(
            train_timestamps,
            train_features,
            normalization_type,
            normalize_timestamps,
        )

        train_timestamps, train_features = self._apply_normalization(
            train_timestamps,
            train_features,
            normalization_type,
            norm_stats,
            normalize_timestamps,
        )
        val_timestamps, val_features = self._apply_normalization(
            val_timestamps,
            val_features,
            normalization_type,
            norm_stats,
            normalize_timestamps,
        )
        test_timestamps, test_features = self._apply_normalization(
            test_timestamps,
            test_features,
            normalization_type,
            norm_stats,
            normalize_timestamps,
        )

        train_rows = LocalDataset(train_timestamps, train_features)
        val_rows = LocalDataset(val_timestamps, val_features)
        test_rows = LocalDataset(test_timestamps, test_features)

        window_cfg = self.config.get("window", {})
        if not isinstance(window_cfg, Mapping):
            raise ValueError("task.config.window must be a mapping")

        stride = int(window_cfg.get("stride", 1))
        context_length = int(window_cfg.get("context_length", 24))

        device = cfg.get("train", {}).get("device", "cpu")
        train_dataset = BasicSlidingWindowDataset(
            train_rows,
            stride=stride,
            context_length=context_length,
            device=device,
        )
        val_dataset = BasicSlidingWindowDataset(
            val_rows,
            stride=stride,
            context_length=context_length,
            device=device,
        )
        test_dataset = BasicSlidingWindowDataset(
            test_rows,
            stride=stride,
            context_length=context_length,
            device=device,
        )

        dataloader_cfg = self.config.get("dataloader", {})
        if not isinstance(dataloader_cfg, Mapping):
            dataloader_cfg = {}

        train_batch_size = int(dataloader_cfg.get("train_batch_size", 64))
        eval_batch_size = int(dataloader_cfg.get("eval_batch_size", train_batch_size))
        num_workers = int(dataloader_cfg.get("num_workers", 0))

        train_dl = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        val_dl = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        test_dl = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        return train_dl, val_dl, test_dl

    @staticmethod
    def visualize(pred: torch.Tensor, batch: dict, outdir: Union[str, Path]) -> None:
        """Visualize prediction and target along with input values."""
        # TODO: add normal/anomal elements visualization on test
        target = batch["x"].detach().cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(target[0].squeeze(), label="Target")
        plt.plot(pred.detach().cpu().numpy()[0].squeeze(), label="Prediction")
        plt.legend()
        plt.title("reconstruction Task Visualization")
        plt.savefig(os.path.join(outdir, "reconstruction_visualization.png"))
        plt.close()
