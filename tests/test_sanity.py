"""Tests for reusable sanity checks.

These tests intentionally use tiny local task/model/dataset doubles instead of
real experiment configs. The goal is to validate the check mechanism itself:

- finite checks fail on non-finite tensor values;
- forecasting window checks catch target/input timestamp mistakes;
- split checks catch leakage across train/val/test time ranges;
- BaseRunner keeps delegating its public check methods to sandbox.utils.check.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sandbox.runners.base_runner import BaseRunner
from sandbox.utils.check import (
    finite_dataset_check,
    off_by_one_check,
    split_check,
)


class DummyTask:
    """Minimal task contract implementation used by runner check tests.

    Real tasks can build dataloaders, metrics, visualizations, and task-specific
    losses. These tests only need stable MSE behavior, so this double keeps the
    surface area deliberately small.
    """

    def loss_fn(self, prediction, batch):
        """Compute MSE against batch targets in the formats checks support."""
        if isinstance(batch, dict):
            target = batch["y"]
        elif isinstance(batch, (list, tuple)) and len(batch) > 1:
            target = batch[1]
        else:
            target = torch.zeros_like(prediction)
        return nn.functional.mse_loss(prediction, target)

    def configure_metrics(self, cfg):
        """Expose the metric contract BaseRunner expects."""
        return {"mse": self.loss_fn}

    def build_dataloaders(self, cfg):
        """Keep dataloader construction out of these focused unit tests."""
        raise NotImplementedError


class TinyLinear(nn.Module):
    """Small model that can overfit one synthetic batch quickly.

    The model implements the runner model protocol (`forward`, `predict`, and
    `configure_optimizers`) without depending on the production MLP module. That
    keeps these tests focused on sanity-check behavior, not architecture.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, batch):
        """Accept tensor, tuple/list, or dict batches like production models."""
        if isinstance(batch, dict):
            x = batch["x"]
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        return self.linear(x)

    def predict(self, batch):
        """Inference alias required by the model protocol."""
        return self(batch)

    def configure_optimizers(self, cfg):
        """Return a simple optimizer fast enough for overfit checks."""
        lr = cfg.get("lr", 0.1)
        if isinstance(cfg.get("train", None), dict):
            lr = cfg["train"].get("lr", lr)
        return torch.optim.SGD(self.parameters(), lr=lr), None

    def save_checkpoint(self, checkpoint_dir):
        pass

    def load_checkpoint(self, checkpoint_dir):
        pass


class TimeWindowDataset(Dataset):
    """Synthetic forecasting windows with explicit timestamp control.

    Values are chosen so timestamps are the data. With `context_length=3` and
    `horizon=2`, a correct first sample is:

    - x_t = [0, 1, 2]
    - y_t = [3, 4]

    `target_shift` intentionally corrupts that alignment for negative tests.
    """

    def __init__(
        self,
        start: int,
        length: int,
        context_length: int = 3,
        horizon: int = 2,
        target_shift: int = 0,
    ):
        self.start = start
        self.length = length
        self.context_length = context_length
        self.horizon = horizon
        self.target_shift = target_shift

    def __len__(self):
        """Return the number of synthetic windows."""
        return self.length

    def __getitem__(self, idx):
        """Build one deterministic window and its timestamp metadata."""
        window_start = self.start + idx
        x_t = torch.arange(
            window_start,
            window_start + self.context_length,
            dtype=torch.float32,
        )
        y_start = window_start + self.context_length + self.target_shift
        y_t = torch.arange(y_start, y_start + self.horizon, dtype=torch.float32)
        return {
            "x": x_t.unsqueeze(-1),
            "y": y_t.unsqueeze(-1),
            "x_t": x_t,
            "y_t": y_t,
        }


def test_finite_dataset_check_detects_non_finite_values():
    """finite_dataset_check should fail fast when any tensor has NaN/Inf."""
    dataloader = DataLoader(TensorDataset(torch.tensor([[1.0], [float("nan")]])))

    with pytest.raises(AssertionError, match="non-finite"):
        finite_dataset_check(dataloader)


def test_off_by_one_check_accepts_contiguous_windows():
    """off_by_one_check should accept y_t that starts right after x_t."""
    dataloader = DataLoader(TimeWindowDataset(start=0, length=4), batch_size=2)

    off_by_one_check(dataloader)


def test_off_by_one_check_rejects_shifted_target_window():
    """off_by_one_check should reject targets that overlap the input context."""
    dataloader = DataLoader(
        TimeWindowDataset(start=0, length=4, target_shift=-1),
        batch_size=2,
    )

    with pytest.raises(AssertionError, match="last x_t timestamp"):
        off_by_one_check(dataloader)


def test_split_check_rejects_overlapping_time_indices():
    """split_check should catch timestamp leakage between train and validation."""
    train_dl = DataLoader(TimeWindowDataset(start=0, length=2), batch_size=2)
    val_dl = DataLoader(TimeWindowDataset(start=2, length=2), batch_size=2)
    test_dl = DataLoader(TimeWindowDataset(start=20, length=2), batch_size=2)

    with pytest.raises(ValueError, match="Overlap found"):
        split_check(train_dl, val_dl, test_dl)


def test_split_check_accepts_disjoint_time_indices():
    """split_check should accept chronological splits with disjoint timestamps."""
    train_dl = DataLoader(TimeWindowDataset(start=0, length=2), batch_size=2)
    val_dl = DataLoader(TimeWindowDataset(start=10, length=2), batch_size=2)
    test_dl = DataLoader(TimeWindowDataset(start=20, length=2), batch_size=2)

    split_check(train_dl, val_dl, test_dl)


def test_base_runner_delegates_sanity_checks(tmp_path):
    """BaseRunner.run_checks should call the shared check mechanism.

    This regression test protects the refactor where check implementations moved
    out of BaseRunner and into sandbox.utils.check. The train dataloader is a
    plain TensorDataset, so time-window checks are intentionally skipped while
    model, finite-data, finite-gradient, output, and overfit checks still run.
    """
    x = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
    y = 2.0 * x + 1.0
    train_dl = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    config = {
        "train": {
            "device": "cpu",
            "epochs": 1,
            "lr": 0.1,
            "early_stopping": {"patience": 1},
        },
        "eval": {"init": {"main_val_metric": "loss", "key_val_metric": "mse"}},
        "test": {},
    }
    runner = BaseRunner(
        config=config,
        logger=None,
        output_dir=tmp_path,
        task=DummyTask(),
        model=TinyLinear(),
    )
    runner.train_dl = train_dl

    results = runner.run_checks()

    assert results
    assert all(result.ok for result in results)
