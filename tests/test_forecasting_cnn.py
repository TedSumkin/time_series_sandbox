"""Tests for the shallow Conv1d forecasting model."""

from __future__ import annotations

import torch
import pytest

from sandbox.models.forecasting_cnn import ForecastingCNN


def test_forecasting_cnn_returns_forecasting_shape():
    model = ForecastingCNN(input_channels=7, output_dim=1, horizon=20)
    batch = {"x": torch.randn(4, 24, 7)}

    prediction = model(batch)

    assert prediction.shape == (4, 20, 1)


def test_forecasting_cnn_rejects_feature_mismatch():
    model = ForecastingCNN(input_channels=7, output_dim=1, horizon=20)
    batch = {"x": torch.randn(4, 24, 6)}

    with pytest.raises(ValueError, match="Input feature size mismatch"):
        model(batch)


def test_forecasting_cnn_checkpoint_roundtrip(tmp_path):
    model = ForecastingCNN(input_channels=7, output_dim=1, horizon=20)
    batch = {"x": torch.randn(2, 24, 7)}

    expected = model.predict(batch)
    model.save_checkpoint(tmp_path)

    loaded = ForecastingCNN(input_channels=7, output_dim=1, horizon=20)
    loaded.load_checkpoint(tmp_path)

    torch.testing.assert_close(loaded.predict(batch), expected)
