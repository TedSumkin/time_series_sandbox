"""Forecasting task, preprocessing, and sliding-window dataset helpers.

This module is responsible for the task-specific part of the pipeline:

- reading the raw row-wise dataset configured under ``cfg.data``
- splitting the time series chronologically into train/val/test
- fitting normalization statistics on the train split only
- converting each split into sliding windows
- exposing loss/metrics utilities expected by ``BaseRunner``

The raw dataset itself is intentionally simple and only provides
``(timestamp, features, target)`` rows. All forecasting semantics live here.
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
        target: TensorLike,
    ) -> None:
        super().__init__()

        self.timestamps = _to_float_tensor(timestamps)
        self.features = _to_float_tensor(features)
        self.target = _to_float_tensor(target)

        if self.timestamps.ndim != 1:
            raise ValueError("timestamps must be a 1D tensor")
        if self.features.ndim != 2:
            raise ValueError("features must be a 2D tensor with shape (N, F)")
        if self.target.ndim != 1:
            raise ValueError("target must be a 1D tensor with shape (N,)")
        if not (len(self.timestamps) == len(self.features) == len(self.target)):
            raise ValueError(
                "timestamps, features, and target must have the same length"
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.timestamps[idx], self.features[idx], self.target[idx]


class BasicSlidingWindowDataset(Dataset):
    """Convert a row-wise dataset into forecasting windows.

    Each sample is returned as a dictionary compatible with the repository
    batch contract:

    - ``x``: input context of shape ``(T, F)``
    - ``y``: prediction target of shape ``(H, 1)``
    - ``x_t``: input timestamps of shape ``(T,)``
    - ``y_t``: target timestamps of shape ``(H,)``
    """

    def __init__(
        self,
        dataset: Dataset,
        stride: int,
        context_length: int,
        horizon: int,
        start_idx: int = 0,
        end_idx: int | None = None,
        part: str = "train",
        device: str = "cpu",
    ) -> None:
        super().__init__()

        if part not in {"train", "val", "test"}:
            raise ValueError("part must be one of 'train', 'val', or 'test'")
        self.part = part
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")

        self.stride = stride
        self.context_length = context_length
        self.horizon = horizon
        self.data = dataset
        self.start_idx = start_idx
        self.end_idx = len(dataset) if end_idx is None else end_idx

        self._configure_windows()
        self.timestamps, self.inputs, self.targets = self._collect_data()
        self.timestamps = self.timestamps.to(device)
        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)

    def _configure_windows(self) -> None:
        last_start = self.end_idx - self.context_length - self.horizon + 1
        if last_start <= self.start_idx:
            raise ValueError(
                "Split is too short to build even one sliding window with the "
                "requested context_length and horizon."
            )

        self.start_indices = torch.arange(
            self.start_idx, last_start, self.stride, dtype=torch.long
        )
        self.target_indices = self.start_indices + self.context_length
        self.end_indices = self.target_indices + self.horizon

    def _collect_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dataset_length = len(self.data)
        if dataset_length == 0:
            raise ValueError(
                "Sliding window dataset cannot be built from an empty dataset."
            )

        timestamps = []
        inputs = []
        targets = []

        for idx in range(dataset_length):
            timestamp, input_tensor, target = self.data[idx]
            timestamps.append(_to_float_tensor(timestamp))
            inputs.append(_to_float_tensor(input_tensor))
            targets.append(_to_float_tensor(target))

        return torch.stack(timestamps), torch.stack(inputs), torch.stack(targets)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = int(self.start_indices[idx].item())
        target_idx = int(self.target_indices[idx].item())
        end_idx = int(self.end_indices[idx].item())

        input_timestamp_window = self.timestamps[start_idx:target_idx]
        target_timestamp_window = self.timestamps[target_idx:end_idx]
        input_window = self.inputs[start_idx:target_idx]
        target_window = self.targets[target_idx:end_idx].unsqueeze(-1)

        return {
            "x": input_window,
            "y": target_window,
            "x_t": input_timestamp_window,
            "y_t": target_timestamp_window,
            "meta": {"split_part": self.part},
        }


class ForecastingTask(nn.Module):
    """Task object that prepares forecasting data and computes objectives."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        self.config = config
        self.loss = self._instantiate_loss()
        self.metrics = self._instantiate_metrics()

    def _loss_name(self) -> str:
        loss_cfg = self.config.get("loss", {})
        if isinstance(loss_cfg, Mapping):
            return str(loss_cfg.get("name", "mse")).lower()
        return str(loss_cfg).lower()

    def _metric_names(self) -> Dict[str, str]:
        metrics_cfg = self.config.get("metrics", {})
        if not isinstance(metrics_cfg, Mapping):
            return {"mse": "mse"}

        metric_names = {}
        for key, value in metrics_cfg.items():
            if isinstance(value, Mapping):
                metric_names[str(key)] = str(value.get("name", key)).lower()
            else:
                metric_names[str(key)] = str(value).lower()
        return metric_names or {"mse": "mse"}

    def _instantiate_loss(self):
        loss_name = self._loss_name()
        if loss_name == "mse":
            return nn.MSELoss()
        if loss_name == "mae":
            return nn.L1Loss()
        raise NotImplementedError(f"Unsupported loss: {loss_name}")

    def _instantiate_metrics(self):
        metric_names = self._metric_names()
        metric_fns = {}

        for metric_key, metric_name in metric_names.items():
            if metric_name == "mse":
                metric_fns[metric_key] = (
                    lambda prediction, batch: nn.functional.mse_loss(
                        prediction, batch["y"]
                    )
                )
            elif metric_name == "mae":
                metric_fns[metric_key] = (
                    lambda prediction, batch: nn.functional.l1_loss(
                        prediction, batch["y"]
                    )
                )
            else:
                raise NotImplementedError(f"Unsupported metric: {metric_name}")
        return metric_fns

    def configure_metrics(self, cfg: Mapping[str, object] | None = None):
        """Return metric callables expected by ``BaseRunner``."""
        return self.metrics

    def loss_fn(self, outputs: torch.Tensor, batch: dict):
        """Compute task loss from model outputs and batch targets."""
        return self.loss(outputs, batch["y"])

    def _instantiate_scaler(self, normalization_type: str):
        """Return a normalization mode identifier after validation."""
        normalization_type = normalization_type.lower()
        if normalization_type not in {"standard", "minmax", "none"}:
            raise NotImplementedError(
                f"Normalization type {normalization_type} is not implemented yet"
            )
        return normalization_type

    def _resolve_split(self) -> Tuple[float, float, float]:
        split_cfg = self.config.get("split", {})
        if not isinstance(split_cfg, Mapping):
            raise ValueError("task.config.split must be a mapping")

        train_ratio = float(split_cfg.get("train", 0.7))
        val_ratio = float(split_cfg.get("val", 0.15))
        test_ratio = float(split_cfg.get("test", 0.15))

        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError("Train/Val/Test split ratios must sum to 1.0")

        return train_ratio, val_ratio, test_ratio

    def _split_indices(self, dataset_length: int) -> Tuple[slice, slice, slice]:
        train_ratio, val_ratio, _ = self._resolve_split()

        train_end = int(dataset_length * train_ratio)
        val_end = int(dataset_length * (train_ratio + val_ratio))

        if train_end <= 0 or val_end <= train_end or val_end >= dataset_length:
            raise ValueError(
                "Chronological split produced an empty train, val, or test partition."
            )

        return (
            slice(0, train_end),
            slice(train_end, val_end),
            slice(val_end, dataset_length),
        )

    def _fit_normalization_stats(
        self,
        train_timestamps: torch.Tensor,
        train_features: torch.Tensor,
        train_targets: torch.Tensor,
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
            stats["target_center"] = train_targets.mean()
            stats["target_scale"] = train_targets.std(unbiased=False).clamp_min(eps)
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
            stats["target_center"] = train_targets.min()
            stats["target_scale"] = (
                train_targets.max() - train_targets.min()
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
        targets: torch.Tensor,
        normalization_type: str,
        stats: Mapping[str, torch.Tensor],
        normalize_timestamps: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if normalization_type == "none":
            return timestamps, features, targets

        if normalization_type in {"standard", "minmax"}:
            norm_features = (features - stats["features_center"]) / stats[
                "features_scale"
            ]
            norm_targets = (targets - stats["target_center"]) / stats["target_scale"]
            norm_timestamps = timestamps

            if normalize_timestamps:
                norm_timestamps = (timestamps - stats["timestamps_center"]) / stats[
                    "timestamps_scale"
                ]
        else:
            raise NotImplementedError(
                f"Unsupported normalization type: {normalization_type}"
            )
        return norm_timestamps, norm_features, norm_targets

    def build_dataloaders(self, cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Build chronological train/val/test dataloaders for forecasting."""
        data_cfg = cfg["data"]
        dataset = instantiate(data_cfg["dataset"])

        timestamps, features, target = DatasetExtractor(dataset).extract_data()
        timestamps = _to_float_tensor(timestamps)
        features = _to_float_tensor(features)
        target = _to_float_tensor(target)

        train_slice, val_slice, test_slice = self._split_indices(len(dataset))

        train_timestamps = timestamps[train_slice]
        train_features = features[train_slice]
        train_targets = target[train_slice]

        val_timestamps = timestamps[val_slice]
        val_features = features[val_slice]
        val_targets = target[val_slice]

        test_timestamps = timestamps[test_slice]
        test_features = features[test_slice]
        test_targets = target[test_slice]

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
            train_targets,
            normalization_type,
            normalize_timestamps,
        )

        train_timestamps, train_features, train_targets = self._apply_normalization(
            train_timestamps,
            train_features,
            train_targets,
            normalization_type,
            norm_stats,
            normalize_timestamps,
        )
        val_timestamps, val_features, val_targets = self._apply_normalization(
            val_timestamps,
            val_features,
            val_targets,
            normalization_type,
            norm_stats,
            normalize_timestamps,
        )
        test_timestamps, test_features, test_targets = self._apply_normalization(
            test_timestamps,
            test_features,
            test_targets,
            normalization_type,
            norm_stats,
            normalize_timestamps,
        )

        train_rows = LocalDataset(train_timestamps, train_features, train_targets)
        val_rows = LocalDataset(val_timestamps, val_features, val_targets)
        test_rows = LocalDataset(test_timestamps, test_features, test_targets)

        window_cfg = self.config.get("window", {})
        if not isinstance(window_cfg, Mapping):
            raise ValueError("task.config.window must be a mapping")

        stride = int(window_cfg.get("stride", 1))
        context_length = int(window_cfg.get("context_length", 24))
        horizon = int(window_cfg.get("horizon", 1))

        device = cfg.get("train", {}).get("device", "cpu")
        train_dataset = BasicSlidingWindowDataset(
            train_rows,
            stride=stride,
            context_length=context_length,
            horizon=horizon,
            device=device,
        )
        val_dataset = BasicSlidingWindowDataset(
            val_rows,
            stride=stride,
            context_length=context_length,
            horizon=horizon,
            device=device,
        )
        test_dataset = BasicSlidingWindowDataset(
            test_rows,
            stride=stride,
            context_length=context_length,
            horizon=horizon,
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
        target = batch["y"].detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(target[0].squeeze(), label="Target")
        plt.plot(pred[0].squeeze(), label="Prediction")
        plt.legend()
        plt.title("Forecasting Task Visualization")
        plt.savefig(os.path.join(outdir, "forecasting_visualization.png"))
        plt.close()
