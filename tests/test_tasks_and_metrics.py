"""Regression tests for task and metric helpers."""

from __future__ import annotations

import torch

from sandbox.metrics.horizon_wise import pointwiseMAE, pointwiseMSE
from sandbox.tasks.anomaly_detection_task import (
    ForecastingAnomalyDetectionTask,
    ReconstructionAnomalyDetectionTask,
)
from sandbox.tasks.reconstruction_task import (
    BasicSlidingWindowDataset,
    LocalDataset,
    ReconstructionTask,
)


def test_pointwise_metrics_keep_input_shape():
    y_pred = torch.randn(2, 3, 4)
    y_true = torch.randn(2, 3, 4)

    assert pointwiseMSE()(y_pred, y_true).shape == y_pred.shape
    assert pointwiseMAE()(y_pred, y_true).shape == y_pred.shape


def test_reconstruction_task_builds_windows_and_metrics_against_x():
    rows = LocalDataset(torch.arange(10.0), torch.randn(10, 2))
    dataset = BasicSlidingWindowDataset(rows, stride=1, context_length=3)
    batch = {"x": torch.randn(2, 3, 2)}
    task = ReconstructionTask(
        {"loss": {"name": "mse"}, "metrics": {"mse": "mse", "mae": "mae"}}
    )

    assert len(dataset) == 7
    assert dataset[0]["x"].shape == (3, 2)
    assert task.loss_fn(batch["x"], batch).shape == torch.Size([])
    assert set(task.configure_metrics({})) == {"mse", "mae"}


def test_anomaly_detection_tasks_accept_missing_val_config():
    cfg = {"loss": {"name": "mse"}, "metrics": {"mse": "mse"}}

    assert ForecastingAnomalyDetectionTask(cfg).anomalous_val is False
    assert ReconstructionAnomalyDetectionTask(cfg).anomalous_val is False
