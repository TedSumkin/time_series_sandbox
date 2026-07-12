"""Anomaly detection task adapters."""

from typing import Any, Mapping

from .forecasting_task import (
    BasicSlidingWindowDataset as ForecastingWindowDataset,
    ForecastingTask,
)
from .reconstruction_task import (
    BasicSlidingWindowDataset as ReconstructionWindowDataset,
    ReconstructionTask,
)


class ForecastingAnomalyDetectionTask(ForecastingTask):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        val_cfg = config.get("val", {})
        self.anomalous_val = bool(val_cfg.get("anomalous_val", False))


class ReconstructionAnomalyDetectionTask(ReconstructionTask):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        val_cfg = config.get("val", {})
        self.anomalous_val = bool(val_cfg.get("anomalous_val", False))


__all__ = [
    "ForecastingAnomalyDetectionTask",
    "ForecastingWindowDataset",
    "ReconstructionAnomalyDetectionTask",
    "ReconstructionWindowDataset",
]
