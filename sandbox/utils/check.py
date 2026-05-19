"""Reusable sanity checks for runner, model, and dataloader contracts."""

from __future__ import annotations
import copy
from dataclasses import dataclass
import gc
import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

from sandbox.contracts import ModelProtocol, TaskProtocol, RunnerProtocol


@dataclass(frozen=True)
class CheckResult:
    """Result object for checks that are run through the helper mechanism."""

    name: str
    ok: bool
    message: str


CheckFn = Callable[[], None]


def run_check(name: str, check_fn: CheckFn) -> CheckResult:
    """Run one check and convert success/failure into a structured result."""
    try:
        check_fn()
    except Exception as exc:
        return CheckResult(name=name, ok=False, message=str(exc))
    return CheckResult(name=name, ok=True, message="passed")


def run_checks(checks: Mapping[str, CheckFn]) -> list[CheckResult]:
    """Run named checks and return structured results."""
    return [run_check(name, check_fn) for name, check_fn in checks.items()]


def _iter_batches(data: DataLoader) -> Iterable[Any]:
    if isinstance(data, Dataset):
        for idx in range(len(data)):
            yield data[idx]
        return
    yield from data


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _first_batch(data: DataLoader) -> Any:
    for batch in _iter_batches(data):
        return batch
    raise ValueError("Cannot inspect an empty dataloader or dataset")


def _assert_outputs_close(
    output_before: Any,
    output_after: Any,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    path: str = "output",
) -> None:
    if isinstance(output_before, torch.Tensor):
        assert isinstance(output_after, torch.Tensor), (
            f"Checkpoint roundtrip failed at {path}: output type changed from "
            f"Tensor to {type(output_after).__name__}"
        )
        assert torch.allclose(
            output_before, output_after, rtol=rtol, atol=atol
        ), f"Checkpoint roundtrip failed at {path}: outputs differ"
        return

    if isinstance(output_before, Mapping):
        assert isinstance(output_after, Mapping), (
            f"Checkpoint roundtrip failed at {path}: output type changed from "
            f"Mapping to {type(output_after).__name__}"
        )
        assert (
            output_before.keys() == output_after.keys()
        ), f"Checkpoint roundtrip failed at {path}: output keys differ"
        for key in output_before:
            _assert_outputs_close(
                output_before[key],
                output_after[key],
                rtol=rtol,
                atol=atol,
                path=f"{path}.{key}",
            )
        return

    if isinstance(output_before, (list, tuple)):
        assert isinstance(
            output_after, type(output_before)
        ), f"Checkpoint roundtrip failed at {path}: output sequence type changed"
        assert len(output_before) == len(
            output_after
        ), f"Checkpoint roundtrip failed at {path}: output sequence length changed"
        for idx, (before_item, after_item) in enumerate(
            zip(output_before, output_after)
        ):
            _assert_outputs_close(
                before_item,
                after_item,
                rtol=rtol,
                atol=atol,
                path=f"{path}[{idx}]",
            )
        return

    assert (
        output_before == output_after
    ), f"Checkpoint roundtrip failed at {path}: outputs differ"


def _metric_to_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return torch.tensor(value, dtype=torch.float32)
    return None


def _set_determinism_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def has_time_window_keys(data: DataLoader) -> bool:
    """Return whether the first batch exposes forecasting time-window keys."""
    batch = _first_batch(data)
    return isinstance(batch, Mapping) and "x_t" in batch and "y_t" in batch


def _as_batched_time_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 1:
        return value.unsqueeze(0)
    if value.ndim == 2:
        return value
    raise ValueError(f"{name} must have shape (T,) or (B, T), got {tuple(value.shape)}")


def _time_keys(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("Time-window checks expect mapping batches with x_t and y_t")
    if "x_t" not in batch or "y_t" not in batch:
        raise KeyError("Time-window checks expect batch keys 'x_t' and 'y_t'")
    x_t = batch["x_t"]
    y_t = batch["y_t"]
    if not isinstance(x_t, torch.Tensor) or not isinstance(y_t, torch.Tensor):
        raise TypeError("Batch keys 'x_t' and 'y_t' must be torch.Tensor values")
    return _as_batched_time_tensor(x_t, "x_t"), _as_batched_time_tensor(y_t, "y_t")


def model_forward_predict_check(model: ModelProtocol) -> None:
    """Check that the model exposes the minimal runner call surface."""
    assert hasattr(model, "forward"), "Model class must have a forward method"
    assert hasattr(model, "predict"), "Model class must have a predict method"
    print("Model forward and predict method check passed")


def finite_dataset_check(
    dataloader: DataLoader,
    dataloader_name: str = "Train",
) -> None:
    """Check that all tensor values emitted by a dataloader are finite."""
    print(f"Checking finite dataset for {dataloader_name} dataloader...")
    for batch in _iter_batches(dataloader):
        for tensor in _iter_tensors(batch):
            assert torch.isfinite(
                tensor
            ).all(), f"{dataloader_name} dataloader contains non-finite values"
    print("Finite dataset check passed")


def finite_model_output_check(
    model: ModelProtocol,
    dataloader: DataLoader,
    device: str | torch.device,
    dataloader_name: str = "Train",
) -> None:
    """Check that model outputs are finite for all batches."""
    print(f"Checking finite model output for {dataloader_name} dataloader...")
    check_model = copy.deepcopy(model)
    check_model.to(device)
    check_model.eval()

    with torch.no_grad():
        for batch in _iter_batches(dataloader):
            output = check_model(batch)
            assert torch.isfinite(
                output
            ).all(), f"{dataloader_name} model output contains non-finite values"

    del check_model
    gc.collect()
    print("Finite model output check passed")


def finite_grad_check(
    model: ModelProtocol,
    task: TaskProtocol,
    dataloader: DataLoader,
    train_config: Mapping[str, Any],
    device: str | torch.device,
) -> None:
    """Check that gradients stay finite before and after one optimizer step."""
    print("Checking finite gradients...")
    check_model = copy.deepcopy(model)
    check_model.train()
    check_model.to(device)

    optimizer, _ = check_model.configure_optimizers(train_config)
    loss_fn = task.loss_fn

    for should_step in (False, True):
        for batch in _iter_batches(dataloader):
            optimizer.zero_grad()
            output = check_model(batch)
            loss = loss_fn(output, batch)
            if isinstance(loss, dict):
                try:
                    loss = loss["total"]
                finally:
                    print("Check your loss dict structure in code")
            loss.backward()

            for param in check_model.parameters():
                if param.grad is not None:
                    assert torch.isfinite(
                        param.grad
                    ).all(), "Model yields non-finite gradients"

            if should_step:
                optimizer.step()

    del check_model
    gc.collect()
    print("Finite gradients check passed")


def overfit_one_batch_check(
    model: ModelProtocol,
    task: TaskProtocol,
    train_dl: DataLoader,
    train_config: Mapping[str, Any],
    device: str | torch.device,
    num_steps: int = 200,
    max_loss_ratio: float = 1e-1,
) -> None:
    """Check that a model can strongly reduce loss on one batch."""
    print("Checking overfitting on one batch...")
    check_model = copy.deepcopy(model)
    check_model.to(device)
    check_model.train()
    optimizer, _ = check_model.configure_optimizers(train_config)
    loss_fn = task.loss_fn

    batch = _first_batch(train_dl)
    losses = []
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = check_model(batch)
        loss = loss_fn(output, batch)
        if isinstance(loss, Mapping):
            loss = loss["total"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    initial_loss = losses[0]
    final_loss = losses[-1]
    if initial_loss == 0.0:
        assert final_loss == 0.0, "Model moved away from zero loss on one batch"
        ratio = 0.0
    else:
        ratio = final_loss / initial_loss

    print(f"Initial loss: {initial_loss}, final loss: {final_loss}, ratio: {ratio}")
    assert ratio < max_loss_ratio, (
        f"Model is not overfitting: final/initial loss ratio {ratio:.4g} "
        f"is not below {max_loss_ratio:.4g}"
    )

    del check_model
    gc.collect()
    print("Overfitting check passed")


def split_check(
    train_dl: DataLoader | Dataset | Iterable[Any],
    val_dl: DataLoader | Dataset | Iterable[Any],
    test_dl: DataLoader | Dataset | Iterable[Any],
) -> None:
    """Check that train, validation, and test time indices do not overlap."""
    split_timestamps = {}
    for split_name, dataloader in (
        ("train", train_dl),
        ("val", val_dl),
        ("test", test_dl),
    ):
        timestamps = []
        for batch in _iter_batches(dataloader):
            x_t, y_t = _time_keys(batch)
            timestamps.extend(x_t.reshape(-1).detach().cpu().tolist())
            timestamps.extend(y_t.reshape(-1).detach().cpu().tolist())
        split_timestamps[split_name] = set(timestamps)

    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        overlap = split_timestamps[left].intersection(split_timestamps[right])
        if overlap:
            raise ValueError(
                f"Overlap found between {left} and {right} timestamps: "
                f"{len(overlap)} shared values"
            )
    print("Split check passed")


def off_by_one_check(
    data: DataLoader,
    dataloader_name: str = "Dataset",
    require_contiguous: bool = False,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> None:
    """Check that forecast target windows start strictly after input windows."""
    print(f"Checking off-by-one errors for {dataloader_name} dataloader...")

    for batch in _iter_batches(data):
        x_t, y_t = _time_keys(batch)

        assert (x_t[:, -1] < y_t[:, 0]).all(), (
            "Off-by-one error detected: last x_t timestamp must be less than "
            "first y_t timestamp"
        )

        if require_contiguous:
            combined = torch.cat([x_t, y_t], dim=1)
            diffs = combined[:, 1:] - combined[:, :-1]
            expected_step = diffs[:, :1]
            assert (expected_step > 0).all(), "Timestamps must be strictly increasing"
            assert torch.allclose(
                diffs, expected_step.expand_as(diffs), rtol=rtol, atol=atol
            ), "Off-by-one error detected: x_t and y_t are not contiguous"

    print("Off-by-one check passed")


def speed_test(
    model: ModelProtocol,
    task: TaskProtocol,
    train_dl: DataLoader,
    train_config: Mapping[str, Any],
    device: str | torch.device,
):
    """Run a simple speed test to ensure the model can process  batches in a reasonable time frame.
    This is not a strict check but can help catch gross inefficiencies.
    """
    import time

    print("Running speed test on one batch...")
    check_model = copy.deepcopy(model)
    check_model.train()
    check_model.to(device)
    optimizer, _ = check_model.configure_optimizers(train_config)
    start_time = time.time()
    print(f"SPEED TEST DEVICE {device}")
    for _ in range(1000):
        batch = next(train_dl.__iter__())
        pred = check_model(batch)
        optimizer.zero_grad()
        loss = task.loss_fn(pred, batch)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Speed test completed in {elapsed_time:.4f} seconds with loss")
    assert elapsed_time < 60.0, "Model forward pass is too slow"
    print("Speed test passed")


def baseline_sanity_check(
    model: ModelProtocol,
    task: TaskProtocol,
    train_config: Mapping[str, Any],
    train_dl: DataLoader,
    val_dl: DataLoader,
    device: str | torch.device,
) -> None:
    """Check that the model does not yield metrics too low in one epoch."""
    print("Running baseline sanity check...")
    check_model = copy.deepcopy(model)
    check_model.train()
    check_model.to(device)
    optimizer, _ = check_model.configure_optimizers(train_config)
    loss_fn = task.loss_fn

    for batch in train_dl:
        pred = check_model(batch)
        optimizer.zero_grad()
        loss = loss_fn(pred, batch)
        loss.backward()
        optimizer.step()

    check_model.eval()
    with torch.no_grad():
        loss_value = 0
        for batch in val_dl:
            pred = check_model(batch)
            loss = loss_fn(pred, batch)
            if isinstance(loss, Mapping):
                loss = loss["total"]
            loss_value += float(loss.detach().cpu())
        print(f"Baseline sanity check loss: {loss_value}")
        assert loss_value < 1e6, "Baseline sanity check failed: loss is too high"
        assert loss_value > 1e-6, "Baseline sanity check failed: loss is too low"
    print("Baseline sanity check passed")


def checkpoint_roundtrip_check(
    model: ModelProtocol,
    device: str | torch.device,
    dataloader: DataLoader,
) -> None:
    """Check that the model can save and load a checkpoint without errors."""
    print("Running checkpoint roundtrip check...")

    # copy model and create first batch output
    check_model = copy.deepcopy(model)
    check_model.to(device)
    check_model.eval()
    first_batch = _first_batch(dataloader)
    with torch.no_grad():
        output_before = check_model.predict(first_batch)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        checkpoint_dir_path = Path(checkpoint_dir)
        check_model.save_checkpoint(checkpoint_dir=checkpoint_dir_path)

        loaded_model = copy.deepcopy(model)
        loaded_model.to(device)
        loaded_model.load_checkpoint(checkpoint_dir=checkpoint_dir_path)
        loaded_model.to(device)
        loaded_model.eval()

        with torch.no_grad():
            output_after = loaded_model.predict(first_batch)

    _assert_outputs_close(output_before, output_after)

    print("Checkpoint roundtrip check passed")


def determinism_check(
    runner: RunnerProtocol,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    seed: int = 0,
    rtol: float = 1e-4,
    atol: float = 1e-6,
):

    print("Starting determinism check")
    base_model = copy.deepcopy(runner.model)
    original_model = runner.model
    original_optimizer = getattr(runner, "optimizer", None)
    original_scheduler = getattr(runner, "scheduler", None)
    original_epochs = getattr(runner, "epochs", None)
    original_output_dir = getattr(runner, "output_dir", None)
    original_checkpoint_dir = getattr(runner, "checkpoint_dir", None)
    original_early_stopping_counter = getattr(runner, "early_stopping_counter", None)
    python_rng_state = random.getstate()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    def run_once(model: ModelProtocol, output_dir: Path) -> Mapping[str, Any]:
        _set_determinism_seed(seed)
        runner.model = model
        runner.epochs = 1
        runner.output_dir = output_dir
        runner.checkpoint_dir = output_dir / "checkpoints"
        runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        runner.early_stopping_counter = 0
        runner.train(train_dl, val_dl)
        runner.model.eval()
        return runner.eval(test_dl, "val")

    try:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            metrics_first = run_once(
                copy.deepcopy(base_model),
                temporary_path / "first",
            )
            metrics_second = run_once(
                copy.deepcopy(base_model),
                temporary_path / "second",
            )
    finally:
        runner.model = original_model
        runner.optimizer = original_optimizer
        runner.scheduler = original_scheduler
        if original_epochs is not None:
            runner.epochs = original_epochs
        if original_output_dir is not None:
            runner.output_dir = original_output_dir
        if original_checkpoint_dir is not None:
            runner.checkpoint_dir = original_checkpoint_dir
        if original_early_stopping_counter is not None:
            runner.early_stopping_counter = original_early_stopping_counter
        random.setstate(python_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    for key, val in metrics_first.items():
        assert key in metrics_second, f"Metric '{key}' missing in second run"
        val_0 = _metric_to_tensor(val)
        val_1 = _metric_to_tensor(metrics_second[key])
        if val_0 is None or val_1 is None:
            continue

        assert torch.allclose(val_0, val_1, rtol=rtol, atol=atol), (
            "Equal starts returned substantially different results: "
            f"{val} and {metrics_second[key]} for key {key}"
        )
    print("Passed determinism check")
    return


def run_default_checks(
    model: ModelProtocol,
    task: TaskProtocol,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    train_config: Mapping[str, Any],
    device: str | torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    runner: RunnerProtocol | None = None,
) -> list[CheckResult]:
    """Run the default runner sanity checks and return structured results."""
    optimizer_config = getattr(runner, "config", train_config)
    checks: dict[str, CheckFn] = {
        "model_forward_predict_check": lambda: model_forward_predict_check(model),
        "overfit_one_batch_check": lambda: overfit_one_batch_check(
            model=model,
            task=task,
            train_dl=train_dl,
            train_config=optimizer_config,
            device=device,
        ),
        "finite_dataset_check": lambda: finite_dataset_check(train_dl, "Train"),
        "finite_grad_check": lambda: finite_grad_check(
            model=model,
            task=task,
            dataloader=train_dl,
            train_config=optimizer_config,
            device=device,
        ),
        "finite_model_output_check_train": lambda: finite_model_output_check(
            model=model,
            dataloader=train_dl,
            device=device,
            dataloader_name="Train",
        ),
        "checkpoint_roundtrip_check": lambda: checkpoint_roundtrip_check(
            model=model,
            device=device,
            dataloader=train_dl,
        ),
    }

    checks["speed_test"] = lambda: speed_test(
        model=model,
        task=task,
        train_dl=train_dl,
        train_config=optimizer_config,
        device=device,
    )

    if val_dl is not None:
        checks["baseline_sanity_check"] = lambda: baseline_sanity_check(
            model=model,
            task=task,
            train_config=optimizer_config,
            train_dl=train_dl,
            val_dl=val_dl,
            device=device,
        )

    if runner is not None and val_dl is not None and test_dl is not None:
        checks["determinism_check"] = lambda: determinism_check(
            runner=runner,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
        )

    if val_dl is not None:
        checks["finite_model_output_check_val"] = lambda: finite_model_output_check(
            model=model,
            dataloader=val_dl,
            device=device,
            dataloader_name="Val",
        )
    if test_dl is not None:
        checks["finite_model_output_check_test"] = lambda: finite_model_output_check(
            model=model,
            dataloader=test_dl,
            device=device,
            dataloader_name="Test",
        )

    if (
        val_dl is not None
        and test_dl is not None
        and has_time_window_keys(train_dl)
        and has_time_window_keys(val_dl)
        and has_time_window_keys(test_dl)
    ):
        checks["split_check"] = lambda: split_check(train_dl, val_dl, test_dl)
        checks["off_by_one_check_train"] = lambda: off_by_one_check(train_dl, "Train")
        checks["off_by_one_check_val"] = lambda: off_by_one_check(val_dl, "Val")
        checks["off_by_one_check_test"] = lambda: off_by_one_check(test_dl, "Test")

    results = run_checks(checks)
    failed = [result for result in results if not result.ok]
    if failed:
        messages = "\n".join(f"{result.name}: {result.message}" for result in failed)
        raise AssertionError(messages)
    return results
