"""Utilities for extracting raw tensors from repository datasets.

This module currently provides a small adapter class that converts a dataset
into three tensors:

- timestamps
- input features
- targets

Why this helper exists
----------------------
Different datasets in the sandbox may expose their content in different ways.
Some datasets already store tensors as public attributes such as
``dataset.data`` or ``dataset.target``. Others may only expose samples through
``__getitem__``. ``DatasetExtractor`` supports both styles:

1. Attribute-based extraction:
   Use this when the dataset already stores full arrays/tensors on itself.
2. Item-based extraction:
   Use this as a fallback when no attribute names are provided. In this mode,
   the extractor iterates through the dataset and expects each sample to follow
   the repository convention ``(timestamp, features, target)``.

The class is intentionally simple and stateful: the constructor stores the
dataset reference and metadata, while ``extract_data`` performs the actual
extraction. This makes it easy to configure once and invoke later inside
preprocessing or normalization code.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetExtractor:
    """Extract timestamp, feature, and target tensors from a dataset.

    The extractor supports two access patterns:

    - direct attribute access, e.g. ``dataset.data`` and ``dataset.target``
    - iterative access through ``dataset[idx]``

    The iterative path is intended as a generic fallback for datasets whose
    samples are already returned in `(timestamp, input_tensor, target)` format.
    The attribute path is more efficient when the dataset already stores its
    entire content as tensors or NumPy arrays.
    """

    def __init__(
        self,
        dataset: Dataset,
        feature_attr: str = None,
        target_attr: str = None,
        timestamp_attr: str = None,
    ):
        """Store dataset references and extraction metadata.

        Args:
            dataset: Dataset instance to be extracted later.
            feature_attr: Name of the dataset attribute containing feature
                values. If omitted, the extractor will fall back to iterating
                over dataset samples.
            target_attr: Name of the dataset attribute containing target
                values.
            timestamp_attr: Name of the dataset attribute containing timestamp
                values.

        """
        self.dataset = dataset
        self.feature_attr = feature_attr
        self.target_attr = target_attr
        self.timestamp_attr = timestamp_attr

    def extract_data(self):
        """Return timestamps, inputs, and targets as torch tensors.

        Returns:
            Tuple ``(timestamps, inputs, targets)`` where each element is a
            ``torch.Tensor``.

        Raises:
            ValueError: If no dataset has been configured beforehand.
            AssertionError: If attribute-based extraction is requested but one
                of the provided attribute names does not exist on the dataset.

        Notes:
            When no attribute names are provided, the extractor assumes that
            every dataset sample is a tuple structured as
            ``(timestamp, input_tensor, target)``.
        """

        if self.dataset is None:
            raise ValueError("Dataset is not provided to the extractor.")

        if (
            self.feature_attr is None
            or self.target_attr is None
            or self.timestamp_attr is None
        ):
            # Fallback mode: we do not know where the full arrays live on the
            # dataset, so we reconstruct them by iterating over all samples.
            # This path is slower, but it is generic and works with any dataset
            # that returns `(timestamp, features, target)` from `__getitem__`.
            print(
                "Feature, target, or timestamp attribute is not specified in Data extractor init function. Returning raw dataset output."
            )
            timestamps = []
            inputs = []
            targets = []
            for idx in range(len(self.dataset)):
                timestamp, input_tensor, target = self.dataset[idx]
                timestamps.append(timestamp)
                inputs.append(input_tensor)
                targets.append(target)

            timestamps = np.stack(timestamps)
            inputs = np.stack(inputs)
            targets = np.stack(targets)
        else:
            # Fast path: extract arrays directly from dataset attributes.
            # This avoids Python-level iteration and is preferable for datasets
            # that already expose raw storage such as `dataset.data`.
            assert hasattr(
                self.dataset, self.feature_attr
            ), f"Dataset does not have the specified feature attribute: {self.feature_attr}"
            assert hasattr(
                self.dataset, self.target_attr
            ), f"Dataset does not have the specified target attribute: {self.target_attr}"
            assert hasattr(
                self.dataset, self.timestamp_attr
            ), f"Dataset does not have the specified timestamp attribute: {self.timestamp_attr}"
            timestamps = getattr(self.dataset, self.timestamp_attr)
            inputs = getattr(self.dataset, self.feature_attr)
            targets = getattr(self.dataset, self.target_attr)

            # Normalization and batching code downstream expects torch tensors.
            # Convert NumPy arrays when needed, but keep existing tensors as-is
            # to avoid unnecessary copies.
            if isinstance(timestamps, np.ndarray):
                timestamps = torch.from_numpy(timestamps)
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs)
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)

        return timestamps, inputs, targets


if __name__ == "__main__":

    class DummyAttributeDataset(Dataset):
        """Dataset exposing full arrays via attributes.

        This mirrors the common "raw CSV already loaded into memory" pattern,
        where the dataset stores tensors or NumPy arrays directly on itself.
        """

        def __init__(self) -> None:
            self.timestamp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            self.data = np.array(
                [
                    [10.0, 11.0],
                    [20.0, 21.0],
                    [30.0, 31.0],
                ],
                dtype=np.float32,
            )
            self.target = np.array([100.0, 200.0, 300.0], dtype=np.float32)

        def __len__(self) -> int:
            return len(self.timestamp)

        def __getitem__(self, idx: int):
            return (
                torch.tensor(self.timestamp[idx]),
                torch.tensor(self.data[idx]),
                torch.tensor(self.target[idx]),
            )

    class DummyIterativeDataset(Dataset):
        """Dataset exposing samples only through `__getitem__`.

        This mirrors the fallback extraction path where the extractor has to
        rebuild full tensors by iterating over dataset rows.
        """

        def __init__(self) -> None:
            self.samples = [
                (
                    torch.tensor(1.0, dtype=torch.float32),
                    torch.tensor([1.0, 2.0], dtype=torch.float32),
                    torch.tensor(10.0, dtype=torch.float32),
                ),
                (
                    torch.tensor(2.0, dtype=torch.float32),
                    torch.tensor([3.0, 4.0], dtype=torch.float32),
                    torch.tensor(20.0, dtype=torch.float32),
                ),
                (
                    torch.tensor(3.0, dtype=torch.float32),
                    torch.tensor([5.0, 6.0], dtype=torch.float32),
                    torch.tensor(30.0, dtype=torch.float32),
                ),
            ]

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            return self.samples[idx]

    print("Running DatasetExtractor self-tests...")

    # Test 1: direct attribute extraction should preserve shapes and convert
    # NumPy arrays to torch tensors.
    attr_dataset = DummyAttributeDataset()
    attr_extractor = DatasetExtractor(
        attr_dataset,
        feature_attr="data",
        target_attr="target",
        timestamp_attr="timestamp",
    )
    timestamps, inputs, targets = attr_extractor.extract_data()
    assert isinstance(timestamps, torch.Tensor)
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert timestamps.shape == (3,)
    assert inputs.shape == (3, 2)
    assert targets.shape == (3,)
    assert torch.allclose(timestamps, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(
        inputs,
        torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]),
    )
    assert torch.allclose(targets, torch.tensor([100.0, 200.0, 300.0]))

    # Test 2: iterative extraction should stack individual samples into full
    # tensors with consistent leading dimension.
    iterative_dataset = DummyIterativeDataset()
    iterative_extractor = DatasetExtractor(iterative_dataset)
    timestamps, inputs, targets = iterative_extractor.extract_data()
    assert timestamps.shape == (3,)
    assert inputs.shape == (3, 2)
    assert targets.shape == (3,)
    assert torch.allclose(timestamps, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(
        inputs,
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    assert torch.allclose(targets, torch.tensor([10.0, 20.0, 30.0]))

    # Test 3: the extractor should fail fast if extraction is attempted before
    # a dataset has been configured.
    try:
        DatasetExtractor(None).extract_data()
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when dataset is not configured.")

    # Test 4: attribute-based extraction should fail with a clear assertion if
    # the requested attribute is missing on the dataset object.
    try:
        DatasetExtractor(
            attr_dataset,
            feature_attr="missing_data",
            target_attr="target",
            timestamp_attr="timestamp",
        ).extract_data()
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError for a missing dataset attribute.")

    print("All DatasetExtractor self-tests passed.")
