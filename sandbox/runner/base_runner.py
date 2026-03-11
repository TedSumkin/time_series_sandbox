import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import copy
import gc

import json
# Allow running this file directly:
# `python sandbox/runner/base_runner.py`
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


from sandbox.contracts import BatchLike, TaskProtocol, ModelProtocol


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
        self.optimizer, self.scheduler = self.model.configure_optimizers(
            self.train_config
        )
        self.metrics = self.task.configure_metrics(self.eval_config)

        self.early_stopping_counter = 0

        self.patience = self.train_config["early_stopping"]["patience"]

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
    
    def save_metrics(self, metric_dict: Dict[str, float], destination_path: Union[str, Path]):
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
        self.optimizer, self.scheduler = self.model.configure_optimizers(
            self.train_config
        )
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
                if hasattr(self.model, "save_checkpoint"):
                    self.model.save_checkpoint(self.checkpoint_dir, epoch, metric_dict)
                    
        return best_metric_dict

    def test(self, test_dl: DataLoader):
        """Evaluate the model on the test set.

        Delegates to `self.eval` with mode "test".

        Args:
            test_dl (DataLoader): Test data loader.
        """
        metric_dict = self.eval(test_dl, "test")
        self.save_metrics(metric_dict, Path(self.output_dir / "results.json"))

    def run_checks(self):
        """Run checks on the model.

        Checks include:
            - Model has a forward method
            - Model is overfitting on one batch
            - Dataset is finite
            - Gradients are finite
        """
        self.model_forward_check()
        self.overfit_one_batch_check(self.train_dl)
        self.finite_dataset_check(self.train_dl, "Train")
        self.finite_grad_check(self.train_dl)

    def model_forward_predict_check(self):
        """Check that the model has a forward method.

        Args:
            model (nn.Module): Model to check.
        """
        assert hasattr(self.model, "forward"), "Model class must have a forward method"
        assert hasattr(self.model, "predict"), "Model class must have a predict method"


    def overfit_one_batch_check(self, train_dl: DataLoader, num_steps: int=200):
        """Check that the model is overfitting on one batch.

        Args:
            train_dl (DataLoader): Training data loader.
            num_steps (int): Number of steps to train.
        """
        print("Checking overfitting on one batch...")
        model = copy.deepcopy(self.model)
        model.to(self.device)
        model.train()
        optimizer, scheduler = model.configure_optimizers(self.train_config)
        loss_fn = model.configure_loss_fn(self.train_config)

        for batch in train_dl:
            break
        loss = []
        for i in range(num_steps):
            optimizer.zero_grad()
            output = self.model(batch)

            loss.append(loss_fn(output, batch))
            loss[-1].backward()
            optimizer.step()

        print(f"Initial loss: {loss[0]}, final loss: {loss[-1]}, ratio: {loss[-1] / loss[0]}")
        assert loss[-1] / loss[0] < 1e-3, "Model is not overfitting"
        
        # force memory deallocation
        del model
        gc.collect()
        print("Overfitting check passed")

    def finite_dataset_check(self, dataloader: DataLoader, dataloder_name: str="Train"):
        """Check that the dataset is finite.

        Args:
            dataloader (DataLoader): Dataloader to check.
            dataloder_name (str): Name of the dataloader.
        """
        print(f"Checking finite dataset for {dataloder_name} dataloader...")
        for batch in dataloader:
            assert torch.isfinite(batch).all(), f"{dataloder_name} dataloader contains non-finite values"

    def finite_grad_check(self, dataloader: DataLoader):
        """"Check that the gradients are finite.

        Args:
            dataloader (DataLoader): Dataloader to check.
        """

        print("Checking finite gradients...")
        model = copy.deepcopy(self.model)
        model.train()
        model.to(self.device)

        optimizer, scheduler = model.configure_optimizers(self.train_config)
        loss_fn = model.configure_loss_fn(self.train_config)
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch)
            loss.backward()

            for param in model.parameters():
                assert torch.isfinite(param.grad).all(), f"Model yields non-finite gradients"

        # imitate first epoch
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch)
            loss.backward()

            for param in model.parameters():
                assert torch.isfinite(param.grad).all(), f"Model yields non-finite gradients"
            optimizer.step()
        print("Finite gradients check passed")
        
    
    def finite_model_output_check(self, dataloader: DataLoader, dataloder_name: str="Train"):
        """Check that the model output is finite.

        Args:
            dataloader (DataLoader): Dataloader to check.
            dataloder_name (str): Name of the dataloader.
        """
        print(f"Checking finite model output for {dataloder_name} dataloader...")
        model = copy.deepcopy(self.model)
        model.to(self.device)
        model.eval()
        for batch in dataloader:
            output = self.model(batch)
            assert torch.isfinite(output).all(), f"{dataloder_name} dataloader contains non-finite values"
        print("Finite model output check passed")


# micro-test for all functions.
if __name__ == "__main__":
    import sys
    import tempfile
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.tensorboard import SummaryWriter
    from pathlib import Path

    # ---- Simple model (tiny MLP) ----
    class TinyMLP(nn.Module):
        def __init__(self, input_dim=4, output_dim=1, horizon=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, horizon * output_dim)
            )
            self.horizon = horizon
            self.output_dim = output_dim

        def forward(self, batch):
            # Assume batch is a tensor of shape (B, input_dim)
            if isinstance(batch, dict):
                x = batch["x"]
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            out = self.net(x)
            return out.view(out.shape[0], self.horizon, self.output_dim)

        def predict(self, batch):
            return self(batch)

        def configure_optimizers(self, cfg):
            lr = cfg.get("lr", 0.001)
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.9
            )
            return optimizer, scheduler

    # ---- Simple task ----
    class DummyTask:
        def loss_fn(self, prediction, batch):
            # Dummy loss: MSE between prediction and a dummy target
            if isinstance(batch, dict):
                target = batch.get("y", torch.zeros_like(prediction))
            elif isinstance(batch, (list, tuple)) and len(batch) > 1:
                target = batch[1]
            else:
                target = torch.zeros_like(prediction)
            return nn.functional.mse_loss(prediction, target)

        def configure_metrics(self, cfg):
            # Return a single metric that computes MAE
            def mae(prediction, batch):
                if isinstance(batch, dict):
                    target = batch.get("y", torch.zeros_like(prediction))
                elif isinstance(batch, (list, tuple)) and len(batch) > 1:
                    target = batch[1]
                else:
                    target = torch.zeros_like(prediction)
                return torch.mean(torch.abs(prediction - target))

            return {"mae": mae}

        def build_dataloaders(self, cfg):
            # Not needed for this test
            raise NotImplementedError()

    # ---- Create dummy data ----
    batch_size = 4
    seq_len = 10
    input_dim = 4
    output_dim = 1
    horizon = 1

    # Random features and targets
    X = torch.randn(batch_size * seq_len, input_dim)
    y = torch.randn(batch_size * seq_len, horizon, output_dim)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---- Configuration ----
    config = {
        "train": {
            "device": "cpu",
            "epochs": 2,
            "lr": 0.01,
            "early_stopping": {"patience": 3},
        },
        "eval": {"init": {"main_val_metric": "loss", "key_val_metric": "mae"}},
        "test": {},
    }

    # ---- Temporary directory for logs ----
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SummaryWriter(log_dir=tmpdir)
        model = TinyMLP(input_dim=input_dim, output_dim=output_dim, horizon=horizon)
        task = DummyTask()
        runner = BaseRunner(
            config=config,
            logger=logger,
            output_dir=tmpdir,
            task=task,
            model=model,
        )

        print("BaseRunner instantiated successfully.")

        # ---- Evaluation test ----
        eval_results = runner.eval(dataloader, "val")
        print(f"Evaluation results: {eval_results}")

        # ---- Training test (just one epoch) ----
        # Note: we need to set checkpoint_dir attribute (used in train)
        runner.checkpoint_dir = Path(tmpdir) / "checkpoints"
        runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_metrics = runner.train(model, dataloader, dataloader)
        print(f"Training completed. Best metrics: {best_metrics}")

        # ---- Test test ----
        runner.test(dataloader)
        print("Test executed.")

        print("All basic tests passed.")
