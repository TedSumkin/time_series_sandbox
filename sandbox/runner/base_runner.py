import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, TypedDict, Mapping
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from pathlib import Path


from abc import abstractmethod, ABC


from sandbox.contracts import BatchLike, TaskProtocol, ModelProtocol


class BaseRunner:
    def __init__(
        self,
        config: Union[Dict, DictConfig],
        logger: SummaryWriter,
        output_dir: Union[Path, str],
        task: TaskProtocol,
        model: ModelProtocol,
    ):

        # make fields corresponding to arguments
        self.output_dir = output_dir
        self.logger = logger
        self.model = model
        self.config = config
        self.task = task

        # Establish device to use in the experiment
        self.device = config["train"]["device"]

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.train_config = self.config["train"]
        self.eval_config = self.config["eval"]
        self.test_config = self.config["test"]

        # initialize loaders
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        self.epochs = self.train_config["epochs"]

        self.model = model

        self.main_val_metric = self.eval_config["init"]["main_val_metric"]

        # annotate optimizer and scheduler

        self.optimizer, self.scheduler = self.model.configure_optimizers(
            self.train_config
        )

    @torch.no_grad()
    def eval(self, dataloader: DataLoader, eval_mode: str):

        if eval_mode not in ["val", "test"]:
            raise ValueError(f"Unexpected evaluation mode: {eval_mode}")
        position = 0 if eval_mode == "test" else 1
        description = "Validation" if eval_mode == "val" else "Test"
        pbar_eval = tqdm(
            dataloader,
            desc=description,
            position=position,
        )
        for batch in pbar_eval:
            prediction = self.model.predict(batch)

    def _train_epoch(self, train_dl: DataLoader):
        pbar_train_epoch = tqdm(train_dl, desc="Train", position=1)

        for batch in pbar_train_epoch:
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.task.loss_fn(outputs, batch)  # type: ignore
            if isinstance(loss, Mapping):
                loss = loss["total"]

            loss.backward()

            self.optimizer.step()
            self.scheduler.step()  # type: ignore

    def _val_epoch(self, val_dl: DataLoader):
        self.eval(val_dl, "val")

    def train(self, model, train_dl, val_dl):

        # setup model
        model.train()
        model.to(self.device)

        # train dataloader is expected to exist
        pbar_upper_level = tqdm(range(self.epochs), desc="Epochs", position=0)
        for epoch in pbar_upper_level:
            self._train_epoch(train_dl)
            self._val_epoch(val_dl)

    def test(self, test_dl: DataLoader):
        self.eval(test_dl, "test")


# micro-test for all functions.
if __name__ == "__main__":
    pass
