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
    pass


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

    def load_state_dict(self, chechpoint_path: Union[str, Path]) -> Any: ...

    def __call__(self, batch: BatchLike) -> torch.Tensor: ...

    def train(self, mode: bool = True) -> Any: ...

    def to(self, device: Union[str, torch.device]) -> Any: ...

    def predict(self, batch: BatchLike) -> torch.Tensor: ...

    def configure_optimizers(
        self, cfg: Mapping[str, Any]
    ) -> Tuple[Optimizer, Optional[LRScheduler]]: ...

    def parameters(self) -> Any: ...
