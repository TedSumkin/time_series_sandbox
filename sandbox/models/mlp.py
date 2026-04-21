"""Baseline MLP model for time-series experiments.

The module provides a small feed-forward network that can consume batches in
multiple shapes (tensor, mapping, or sequence) and emits predictions with a
fixed `(B, horizon, output_dim)` layout.
"""

from typing import Mapping, Sequence, Tuple, Union

from pathlib import Path
import torch
from torch import nn


BatchLike = Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]]


class MLP(nn.Module):
    """A simple multi-layer perceptron for fixed-size forecasting outputs.

    The class is intentionally minimal: it supports several batch containers,
    validates dimensions aggressively, and exposes helper methods expected by
    the repository contract (`forward`, `predict`, `configure_optimizers`).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int = 1,
        hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.0,
    ) -> None:
        """Construct an MLP with configurable hidden layers and output horizon.

        Args:
            input_dim: Number of input features after flattening.
            output_dim: Number of target features predicted per horizon step.
            horizon: Number of future steps to predict.
            hidden_dims: Width of hidden MLP layers.
            dropout: Dropout probability inserted after each hidden activation.

        Raises:
            ValueError: If any mandatory dimension argument is not positive.
        """
        super().__init__()
        # Validate basic shape-related hyperparameters early, so configuration
        # mistakes fail immediately and with explicit messages.
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")

        # Persist dimensions because they are reused in runtime checks and
        # output reshaping logic.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon

        # Build an MLP head that maps `input_dim` to `horizon * output_dim`.
        # The final linear layer is reshaped later in `forward`.
        dims = [input_dim, *hidden_dims, horizon * output_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            # Dropout is optional and only added when requested.
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def _extract_x(self, batch: BatchLike) -> torch.Tensor:
        """Extract the model input tensor from supported batch containers.

        Supported input forms:
            1) Plain tensor: interpreted directly as `x`.
            2) Mapping: requires `batch["x"]`.
            3) Sequence: first element is treated as `x`.

        Args:
            batch: A tensor-like object returned by dataset/dataloader/task.

        Returns:
            The extracted input tensor.

        Raises:
            KeyError: If a mapping batch does not contain key `"x"`.
            TypeError: If batch type is unsupported or sequence[0] is not tensor.
        """
        # Fast-path: the dataloader already returns x as a plain tensor.
        if isinstance(batch, torch.Tensor):
            return batch
        # Common task-contract path: batch is a dictionary with named fields.
        if isinstance(batch, Mapping):
            if "x" not in batch:
                raise KeyError("Batch mapping must contain key 'x'")
            return batch["x"]
        # Compatibility path for tuple/list batches, e.g. (x, y, ...).
        if isinstance(batch, Sequence) and len(batch) > 0:
            candidate = batch[0]
            if not isinstance(candidate, torch.Tensor):
                raise TypeError(
                    "First element of batch sequence must be a torch.Tensor"
                )
            return candidate
        # Reject unknown formats explicitly; silent fallback here is dangerous.
        raise TypeError("Unsupported batch type")

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
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
        if x.ndim == 1:
            x = x.unsqueeze(0)
        # Flatten temporal/extra dimensions so MLP receives a feature vector.
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        # Guard against unsupported rank after preprocessing.
        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor after preprocessing, got shape={tuple(x.shape)}"
            )
        # Enforce static input width to prevent accidental shape drift.
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input feature size mismatch: expected {self.input_dim}, got {x.shape[1]}"
            )
        return x

    def forward(self, batch: BatchLike) -> torch.Tensor:
        """Run a forward pass and return predictions in task-friendly layout.

        Args:
            batch: Input batch in one of the supported formats.

        Returns:
            Tensor of shape `(B, horizon, output_dim)`.
        """
        # Step 1: isolate `x` from a potentially rich batch container.
        x = self._extract_x(batch)
        # Step 2: flatten/validate so the MLP sees `(B, input_dim)`.
        x = self._prepare_input(x)
        # Step 3: map features to `horizon * output_dim` raw output.
        pred = self.net(x)
        # Step 4: restore semantic dimensions expected by forecasting code.
        return pred.view(pred.shape[0], self.horizon, self.output_dim)

    def predict(self, batch: BatchLike) -> torch.Tensor:
        """Inference-style alias for the model call.

        Uses `self(batch)` instead of directly calling `forward` so PyTorch
        call-path features (hooks/wrappers) remain active.

        Args:
            batch: Input batch in one of the supported formats.

        Returns:
            Tensor of shape `(B, horizon, output_dim)`.
        """
        return self(batch)

    def configure_optimizers(self, cfg) -> Tuple[torch.optim.Optimizer, None]:
        """Create an Adam optimizer from repository config conventions.

        Expected configuration layout:
            - `cfg.optimizer.lr` and `cfg.optimizer.weight_decay`
            - fallback for lr: `cfg.train.lr` then default `1e-3`
            - fallback for weight_decay: default `0.0`

        Args:
            cfg: Mapping-like experiment config (dict, DictConfig, etc.).

        Returns:
            Tuple `(optimizer, scheduler)` where scheduler is `None`.
        """
        # Read optimizer-related section with defensive defaults.
        optimizer_cfg = {}
        if isinstance(cfg, Mapping):
            optimizer_cfg = cfg.get("optimizer", {}) or {}
            train_cfg = cfg.get("train", {}) or {}
        else:
            optimizer_cfg = cfg.get("optimizer", {})
            train_cfg = cfg.get("train", {})

        # Prefer optimizer-specific lr, then training section fallback.
        lr = float(optimizer_cfg.get("lr", train_cfg.get("lr", 1e-3)))
        weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

        # Baseline choice: Adam without scheduler for simplicity.
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        return optimizer, None

    def save_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        """Save the model checkpoint to the specified directory."""
        torch.save(self.state_dict(), Path(checkpoint_dir) / "MLP.pth")

    def load_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        """Load the model checkpoint from the specified directory."""
        self.load_state_dict(torch.load(Path(checkpoint_dir) / "MLP.pth"))
