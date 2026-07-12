"""Baseline MLP model for time-series experiments.

The module provides a small feed-forward network that can consume batches in
multiple shapes (tensor, mapping, or sequence) and emits predictions with a
fixed `(B, horizon, output_dim)` layout.
"""

from typing import Mapping, Sequence, Tuple, Union

from pathlib import Path
import torch
from torch import nn

from abc import abstractmethod


BatchLike = Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]]


class BaseModel(nn.Module):
    """Simple template for models"""

    @abstractmethod
    def __init__(
        self,
    ) -> None: ...

    @abstractmethod
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def forward(self, batch: BatchLike) -> torch.Tensor: ...

    @abstractmethod
    def predict(self, batch: BatchLike) -> torch.Tensor: ...

    @abstractmethod
    def configure_optimizers(self, cfg) -> Tuple[torch.optim.Optimizer, None]: ...

    @abstractmethod
    def save_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None: ...

    @abstractmethod
    def load_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None: ...
