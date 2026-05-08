import sys
from pathlib import Path
from typing import Dict, Mapping, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import json

# Allow running this file directly:
# `python sandbox/runner/base_runner.py`
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from sandbox.contracts import TaskProtocol, ModelProtocol
from sandbox.utils import check as sanity_check


class BaseRunner:
    """Base runner for training, validation, and testing of forecasting models.

    This class orchestrates the training loop, evaluation, and testing procedures
    using a given model and task protocol. It handles device placement, optimizer
    configuration, metric tracking, and logging.

    Attributes:
        output_dir (Union[Path, str]): Directory for experiment outputs.
        logger (SummaryWriter): TensorBoard logger for metrics.
        model (ModelProtocol): The forecasting model to train/evaluate.
        config (Union[Dict, DictConfig]): Experiment configuration.
        task (TaskProtocol): Task defining loss, metrics, and data semantics.
        device (str): Compute device ('cuda' or 'cpu').
        train_config (dict): Sub‑configuration for training.
        eval_config (dict): Sub‑configuration for evaluation.
        test_config (dict): Sub‑configuration for testing.
        train_dl (DataLoader): Training dataloader (set after initialization).
        val_dl (DataLoader): Validation dataloader (set after initialization).
        test_dl (DataLoader): Test dataloader (set after initialization).
        epochs (int): Number of training epochs.
        main_val_metric (str): Name of the primary validation metric.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler (may be None).
        metrics (dict): Dictionary of metric callables.
    """

    def __init__(
        self,
        config: Union[Dict, DictConfig],
        logger: SummaryWriter,
        output_dir: Union[Path, str],
        task: TaskProtocol,
        model: ModelProtocol,
    ):
        """Initialize the runner with configuration, logging, task, and model.

        Args:
            config (Union[Dict, DictConfig]): Experiment configuration dictionary.
                Must contain "train", "eval", "test" sections.
            logger (SummaryWriter): TensorBoard writer for logging metrics.
            output_dir (Union[Path, str]): Directory where outputs (checkpoints,
                logs, etc.) will be stored.
            task (TaskProtocol): Task object providing loss function, metrics,
                and data semantics.
            model (ModelProtocol): Model object with forward, predict, and
                configure_optimizers methods.
            sanity_run_flag (bool): flag of sanity run mode.

        Raises:
            KeyError: If required configuration keys are missing.
        """

        # make fields corresponding to arguments
        self.output_dir = output_dir
        self.checkpoint_dir = Path(output_dir) / "checkpoints"
        self.logger = logger
        self.model = model
        self.config = config
        self.task = task
        # Establish device to use in the experiment
        self.device = config["train"]["device"]

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(self.device, str) and self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                self.device = "cpu"

        self.train_config = self.config["train"]
        self.eval_config = self.config["eval"]
        self.test_config = self.config["test"]

        # initialize loaders
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        self.epochs = self.train_config["epochs"]

        self.main_val_metric = self.eval_config["init"]["main_val_metric"]

        # annotate optimizer and scheduler
        self.optimizer, self.scheduler = self.model.configure_optimizers(self.config)
        self.metrics = self.task.configure_metrics(self.eval_config)

        self.early_stopping_counter = 0

        self.patience = self.train_config["early_stopping"]["patience"]

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def eval(self, dataloader: DataLoader, eval_mode: str) -> Dict[str, float]:
        """Evaluate the model on a given dataloader.

        Runs inference without gradient computation, computes the task loss
        and all configured metrics averaged over the whole dataset.

        Args:
            dataloader (DataLoader): DataLoader providing evaluation batches.
            eval_mode (str): Either "val" for validation or "test" for testing.
                Determines progress bar positioning and description.

        Returns:
            Dict[str, float]: Dictionary containing "loss" and any other metrics.

        Raises:
            ValueError: If `eval_mode` is not "val" or "test".
        """

        if eval_mode not in ["val", "test"]:
            raise ValueError(f"Unexpected evaluation mode: {eval_mode}")
        position = 0 if eval_mode == "test" else 1
        description = "Validation" if eval_mode == "val" else "Test"
        pbar_eval = tqdm(
            dataloader,
            desc=description,
            position=position,
        )
        total_loss = 0.0
        metric_sums = {name: 0.0 for name in self.metrics}
        num_batches = 0

        for batch in pbar_eval:
            prediction = self.model.predict(batch)
            loss = self.task.loss_fn(prediction, batch)
            if isinstance(loss, Mapping):
                loss = loss["total"]
            total_loss += loss.item()

            # compute metrics
            for name, metric_fn in self.metrics.items():
                metric_sums[name] += metric_fn(prediction, batch).item()

            num_batches += 1

        if num_batches == 0:
            return {"loss": 0.0}

        avg_loss = total_loss / num_batches
        result = {"loss": avg_loss}
        for name in self.metrics:
            result[name] = metric_sums[name] / num_batches
        return result

    def _train_epoch(self, train_dl: DataLoader):
        """Perform a single training epoch.

        Iterates over the training dataloader, computes loss, backpropagates,
        and updates model parameters via the configured optimizer and scheduler.

        Args:
            train_dl (DataLoader): Training data loader.
        """
        pbar_train_epoch = tqdm(train_dl, desc="Train", position=1)

        for batch in pbar_train_epoch:
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.task.loss_fn(outputs, batch)  # type: ignore
            if isinstance(loss, Mapping):
                loss = loss["total"]

            loss.backward()

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

    def _val_epoch(self, val_dl: DataLoader) -> Dict[str, float]:
        """Run validation for one epoch.

        Delegates to `self.eval` with mode "val".

        Args:
            val_dl (DataLoader): Validation data loader.

        Returns:
            Dict[str, float]: Validation metrics.
        """
        return self.eval(val_dl, "val")

    def _early_stopping(
        self, old_metric_dict: Dict[str, float], metric_dict: Dict[str, float]
    ) -> bool:
        if metric_dict[self.key_val_metric] < old_metric_dict[self.key_val_metric]:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.patience:
                return True
        return False

    def save_metrics(
        self, metric_dict: Dict[str, float], destination_path: Union[str, Path]
    ):
        """Save metrics to a file.

        Args:
            metric_dict (Dict[str, float]): Metrics to save.
            destination_path (Union[str, Path]): Path to save metrics to.
        """
        with open(destination_path, "w") as f:
            json.dump(metric_dict, f)

    def train(self, model, train_dl, val_dl):
        """Train the model for a configured number of epochs.

        Sets the model to training mode, moves it to the appropriate device,
        and runs alternating training and validation epochs.

        Args:
            model: The model to train (should be same as self.model).
            train_dl (DataLoader): Training data loader.
            val_dl (DataLoader): Validation data loader.
        """

        # setup model
        model.train()
        model.to(self.device)
        self.key_val_metric = self.eval_config["init"]["key_val_metric"]
        # Re-initialize optimizer and scheduler (optional, could reuse existing)
        self.optimizer, self.scheduler = self.model.configure_optimizers(self.config)
        key_val_metric_value = torch.inf

        # train dataloader is expected to exist
        pbar_upper_level = tqdm(range(self.epochs), desc="Epochs", position=0)

        best_metric_dict = {key: torch.inf for key in self.metrics}
        old_metric_dict = {self.key_val_metric: torch.inf}
        for epoch in pbar_upper_level:
            self._train_epoch(train_dl)
            metric_dict = self._val_epoch(val_dl)

            # check if early stopping is needed
            if self._early_stopping(old_metric_dict, metric_dict):
                pbar_upper_level.close()
                break

            # save best metric dict and checkpoints
            if metric_dict[self.key_val_metric] < key_val_metric_value:
                key_val_metric_value = metric_dict[self.key_val_metric]
                old_metric_dict = best_metric_dict
                best_metric_dict = metric_dict
                # temporal placeholder for checkpoint saving to the output dir
                # TODO: implement save_checkpoint and save_vaL_metrics
                self.model.save_checkpoint(self.checkpoint_dir)
        self.save_metrics(best_metric_dict, Path(self.output_dir) / "val_metrics.json")

        return best_metric_dict

    def load_best_checkpoint(self):
        """Load the best checkpoint from the checkpoint directory."""
        self.model.load_checkpoint(self.checkpoint_dir)

    def test(self, test_dl: DataLoader):
        """Evaluate the model on the test set.

        Delegates to `self.eval` with mode "test".

        Args:
            test_dl (DataLoader): Test data loader.
        """
        metric_dict = self.eval(test_dl, "test")
        self.save_metrics(metric_dict, Path(self.output_dir) / "results.json")
        return metric_dict

    def run_checks(self):
        """Run checks on the model.

        Checks include:
            - Model has a forward method
            - Model is overfitting on one batch
            - Dataset is finite
            - Gradients are finite
        """
        if self.train_dl is None:
            raise ValueError("train_dl must be set before running checks")
        if self.val_dl is None:
            raise ValueError("val_dl must be set before running checks")
        if self.test_dl is None:
            raise ValueError("test_dl must be set before running checks")

        return sanity_check.run_default_checks(
            model=self.model,
            task=self.task,
            train_dl=self.train_dl,
            val_dl=self.val_dl,
            test_dl=self.test_dl,
            train_config=self.train_config,
            device=self.device,
        )
