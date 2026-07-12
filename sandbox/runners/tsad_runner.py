import sys
from pathlib import Path
from typing import Dict, Mapping, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import json

from sandbox.runners.base_runner import BaseRunner

# Allow running this file directly:
# `python sandbox/runner/base_runner.py`
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from sandbox.contracts import TaskProtocol, ModelProtocol
from sandbox.utils import check as sanity_check


class BaseTSADRunner(BaseRunner):
    """Base runner for training, validation, and testing of TSAD models.

    This class orchestrates the training loop, evaluation, and testing procedures
    using a given model and task protocol. It handles device placement, optimizer
    configuration, metric tracking, and logging.

    Attributes:
        output_dir (Union[Path, str]): Directory for experiment outputs.
        logger (SummaryWriter): TensorBoard logger for metrics.
        model (ModelProtocol): The TSAD model to train/evaluate.
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
        super(BaseTSADRunner, self).__init__(config, logger, output_dir, task, model)

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
        # TODO: it's just a placeholder, add functionality later
        return super(BaseTSADRunner, self).eval(dataloader, eval_mode)

    def _early_stopping(
        self, old_metric_dict: Dict[str, float], metric_dict: Dict[str, float]
    ) -> bool:
        # TODO: rewrite metric results
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

    def train(self, train_dl, val_dl):
        """Train the model for a configured number of epochs.

        Sets the model to training mode, moves it to the appropriate device,
        and runs alternating training and validation epochs.

        Args:
            model: The model to train (should be same as self.model).
            train_dl (DataLoader): Training data loader.
            val_dl (DataLoader): Validation data loader.
        """

        # setup model
        self.model.train()
        self.model.to(self.device)
        self.key_val_metric = self.eval_config["init"]["key_val_metric"]
        # Re-initialize optimizer and scheduler (optional, could reuse existing)
        self.optimizer, self.scheduler = self.model.configure_optimizers(self.config)
        key_val_metric_value = torch.inf

        # train dataloader is expected to exist
        pbar_upper_level = tqdm(range(self.epochs), desc="Epochs", position=0)

        best_metric_dict = {key: torch.inf for key in self.metrics}
        old_metric_dict = {self.key_val_metric: torch.inf}

        # check if the model has trainable params

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
                # implement save_checkpoint and save_vaL_metrics
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
            runner=self,
            model=self.model,
            task=self.task,
            train_dl=self.train_dl,
            val_dl=self.val_dl,
            test_dl=self.test_dl,
            train_config=self.train_config,
            device=self.device,
            optimizer=self.optimizer,
        )
