import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from abc import abstractmethod


class BaseRunner:
    def __init__(
        self,
        config: Union[Dict, DictConfig],
        logger: SummaryWriter,
        output_dir: Union[Path, str],
    ):
        self.output_dir = output_dir
        self.logger = logger

        self.config = config
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

    @abstractmethod
    def train(
        self,
    ):
        pass

    @abstractmethod
    def eval(
        self,
    ):
        pass

    @abstractmethod
    def test(
        self,
    ):
        pass


# micro-test for all functions.
if __name__ == "__main__":
    pass
