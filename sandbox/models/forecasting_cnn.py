"""Small Conv1d forecasting model.

The model follows the same repository contract as the MLP:

- accepts rich batches with an ``x`` tensor;
- consumes forecasting windows shaped ``(B, T, F)``;
- returns predictions shaped ``(B, horizon, output_dim)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Tuple, Union

import torch
from torch import nn


BatchLike = Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]]


class ForecastingCNN(nn.Module):
    """A shallow Conv1d baseline for fixed-horizon forecasting."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        horizon: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError("input_channels must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be > 0")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if dropout < 0:
            raise ValueError("dropout must be >= 0")

        self.input_channels = input_channels
        self.output_dim = output_dim
        self.horizon = horizon

        layers = []
        in_channels = input_channels
        padding = kernel_size // 2

        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = hidden_channels

        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_channels, horizon * output_dim)

    def _extract_x(self, batch: BatchLike) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            return batch
        if isinstance(batch, Mapping):
            if "x" not in batch:
                raise KeyError("Batch mapping must contain key 'x'")
            return batch["x"]
        if isinstance(batch, Sequence) and len(batch) > 0:
            candidate = batch[0]
            if not isinstance(candidate, torch.Tensor):
                raise TypeError("First element of batch sequence must be a torch.Tensor")
            return candidate
        raise TypeError("Unsupported batch type")

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to Conv1d layout ``(B, F, T)``."""
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.ndim == 2:
            x = x.unsqueeze(0)

        if x.ndim != 3:
            raise ValueError(
                f"Expected input tensor with shape (B, T, F), got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.input_channels:
            raise ValueError(
                "Input feature size mismatch: "
                f"expected {self.input_channels}, got {x.shape[-1]}"
            )

        return x.transpose(1, 2)

    def forward(self, batch: BatchLike) -> torch.Tensor:
        x = self._extract_x(batch)
        x = self._prepare_input(x)

        features = self.backbone(x)
        pooled = self.pool(features).squeeze(-1)
        pred = self.head(pooled)

        return pred.view(pred.shape[0], self.horizon, self.output_dim)

    def predict(self, batch: BatchLike) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self, cfg) -> Tuple[torch.optim.Optimizer, None]:
        if isinstance(cfg, Mapping):
            optimizer_cfg = cfg.get("optimizer", {}) or {}
            train_cfg = cfg.get("train", {}) or {}
        else:
            optimizer_cfg = cfg.get("optimizer", {})
            train_cfg = cfg.get("train", {})

        lr = float(optimizer_cfg.get("lr", train_cfg.get("lr", 1e-3)))
        weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        return optimizer, None

    def save_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        torch.save(self.state_dict(), Path(checkpoint_dir) / "ForecastingCNN.pth")

    def load_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        self.load_state_dict(
            torch.load(Path(checkpoint_dir) / "ForecastingCNN.pth", map_location="cpu")
        )
