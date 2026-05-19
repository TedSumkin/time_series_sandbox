"""Shared typing contracts used across the sandbox package."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from pathlib import Path

BatchLike = Union[torch.Tensor, Mapping[str, Any], Sequence[Any]]


class RunnerProtocol(Protocol):
    @torch.no_grad()
    def eval(self, dataloader: DataLoader, eval_mode: str) -> Dict[str, float]: ...

    def _train_epoch(self, train_dl: DataLoader) -> Any: ...

    def _val_epoch(self, val_dl: DataLoader) -> Dict[str, float]: ...

    def save_metrics(
        self, metric_dict: Dict[str, float], destination_path: Union[str, Path]
    ) -> None: ...

    def train(self, train_dl, val_dl) -> None: ...

    def load_best_checkpoint(self) -> None: ...

    def test(self, test_dl: DataLoader) -> Union[Mapping, Dict]: ...

    def run_checks(self) -> dict[str, Any]: ...


class TaskProtocol(Protocol):
    """Task interface expected by runners."""

    def loss_fn(self, outputs: Any, batch: BatchLike) -> torch.Tensor: ...

    def configure_metrics(self, cfg: Mapping[str, Any]) -> Dict[str, Any]: ...

    def build_dataloaders(
        self, cfg: Mapping[str, Any]
    ) -> Tuple[DataLoader, DataLoader, DataLoader]: ...

    def visualize(self, prediction, batch, output_dir) -> None: ...


class ModelProtocol(Protocol):
    """Model interface expected by runners."""

    def eval(self) -> Any: ...

    def state_dict(self) -> Any: ...

    def __call__(self, batch: BatchLike) -> torch.Tensor: ...

    def train(self, mode: bool = True) -> Any: ...

    def to(self, device: Union[str, torch.device]) -> Any: ...

    def predict(self, batch: BatchLike) -> torch.Tensor: ...

    def configure_optimizers(
        self, cfg: Mapping[str, Any]
    ) -> Tuple[Optimizer, Optional[LRScheduler]]: ...

    def parameters(self) -> Any: ...

    def save_checkpoint(self, checkpoint_dir: Union[str, Path]) -> Any: ...

    def load_checkpoint(self, checkpoint_dir: Union[str, Path]) -> Any: ...
