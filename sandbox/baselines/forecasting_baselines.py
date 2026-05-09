import torch
from torch import nn

from typing import Tuple, Union
from pathlib import Path


class NaiveLastValueModel(nn.Module):

    def __init__(self, horizon: int):
        super(NaiveLastValueModel, self).__init__()
        self.horizon = horizon
        # self.scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        # self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def _extract_x(self, batch) -> torch.Tensor:
        x = batch["x"]
        return x

    def _prepare_input(self, x: torch.Tensor):
        """Normalize input tensor shape to `(B, input_dim)` for MLP processing.

        Shape handling strategy:
            - 1D tensor is interpreted as one sample and gets batch dimension.
            - Tensors with rank > 2 are flattened from dimension 1 onward.
            - The result must be exactly 2D and match configured `input_dim`.

        Args:
            x: Raw input tensor extracted from the batch.

        Returns:
            A 2D tensor with shape `(batch_size, input_dim)`.

        Raises:
            ValueError: If tensor cannot be converted to the expected shape.
        """
        # Convert a single feature vector to a one-item batch.
        if x.ndim not in [1, 2, 3]:
            raise ValueError(
                f"Expected input tensor to have 1, 2 or 3 input dimensions."
            )
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(-1)

        if x.ndim == 2:
            x = x.unsqueeze(-1)
        # Guard against unsupported rank after preprocessing.
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor after preprocessing, got shape={tuple(x.shape)}"
            )

        return x

    def forward(self, batch):

        x = self._extract_x(batch)
        x = self._prepare_input(x)

        y = x[:, -1:, -1:]
        y = y.expand((y.shape[0], self.horizon, y.shape[-1]))
        y = y  # * self.scale + self.bias
        return y

    def predict(self, batch):
        return self(batch)

    def configure_optimizers(self, cfg) -> Tuple[torch.optim.Optimizer, None]:

        class NotAnOptimizer:
            def __init__(self, cfg):
                self.cfg = cfg

            def step(self):
                return

            def zero_grad(self):
                return

        return NotAnOptimizer(cfg), None
        # return torch.optim.AdamW(lr=0.1, params=self.parameters()), None

    def save_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        """Save the model checkpoint to the specified directory."""
        torch.save(
            self.state_dict(), Path(checkpoint_dir) / "NaiveBaselineForOTOnly.pth"
        )

    def load_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        """Load the model checkpoint from the specified directory."""
        self.load_state_dict(
            torch.load(Path(checkpoint_dir) / "NaiveBaselineForOTOnly.pth")
        )


class EWMA(nn.Module):
    def __init__(self, alpha: float, horizon: int):
        super(EWMA, self).__init__()
        self.alpha = alpha
        self.horizon = horizon
        print(
            "EWMA initialized, using only the last value of the input sequence for forecasting."
        )

    def _extract_x(self, batch) -> torch.Tensor:
        x = batch["x"]
        return x

    def _prepare_input(self, x: torch.Tensor):

        if x.ndim not in [1, 2, 3]:
            raise ValueError(
                f"Expected input tensor to have 1, 2 or 3 input dimensions."
            )
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(-1)

        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor after preprocessing, got shape={tuple(x.shape)}"
            )

        return x

    def forward(self, batch):

        x = self._extract_x(batch)
        x = self._prepare_input(x)
        series = x[:, :, -1:]
        s = series[:, 0:1, :]

        for t in range(1, series.shape[1]):
            s = self.alpha * series[:, t : t + 1, :] + (1 - self.alpha) * s

        pred = s.expand((s.shape[0], self.horizon, s.shape[-1]))
        return pred

    def predict(self, batch):
        return self(batch)

    def configure_optimizers(self, cfg) -> Tuple[torch.optim.Optimizer, None]:

        class NotAnOptimizer:
            def __init__(self, cfg):
                self.cfg = cfg

            def step(self):
                return

            def zero_grad(self):
                return

        return NotAnOptimizer(cfg), None
        # return torch.optim.AdamW(lr=0.1, params=self.parameters()), None

    def save_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        """Save the model checkpoint to the specified directory."""
        torch.save(
            self.state_dict(), Path(checkpoint_dir) / "NontrainableEWMAForOTOnly.pth"
        )

    def load_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        """Load the model checkpoint from the specified directory."""
        self.load_state_dict(
            torch.load(Path(checkpoint_dir) / "NontrainableEWMAForOTOnly.pth")
        )


if __name__ == "__main__":
    model = NaiveLastValueModel(horizon=5)
    print(list(model.parameters()))
