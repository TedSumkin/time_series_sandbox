"""This file is an entrypoint for the project."""

# run.py
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from sandbox.contracts import TaskProtocol

from typing import Union


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:

    # get experiment name and experiment mode
    hydra_config = HydraConfig.get()
    experiment_mode = hydra_config.mode.name.lower()

    experiment_name = cfg["experiment_name"]

    print(experiment_name, experiment_mode)
    if experiment_mode == "multirun":
        if hydra_config.job.num == 0:
            print(f"Starting multirun in {experiment_name}")

    # set hydra output dir
    out_dir = Path(hydra_config.runtime.output_dir)
    logger = SummaryWriter(log_dir=str(out_dir / "tb"))

    # Create key objects for the experiment: model, task, runner, etc.

    task = instantiate(cfg.tasks)
    model = instantiate(cfg.models)

    # runner = instantiate(
    #     cfg.runner,
    #     logger=logger,
    #     output_dir=out_dir,
    #     model=model,
    #     task=task,
    # )

    # # Pipeline orchestration
    # train_dl, val_dl, test_dl = task.build_dataloaders(cfg)
    # runner.train(model=model, train_dl=train_dl, val_dl=val_dl)
    # runner.eval(dataloader=val_dl, eval_mode="val")
    # runner.test(model=model, test_dl=test_dl)

    if experiment_mode == "multirun":
        return "multirun", hydra_config.sweep.dir  # type: ignore
    else:
        return "single", hydra_config.runtime.output_dir  # type: ignore


def find_best_subdir(dir_path: Union[Path, str]) -> Path:
    pass


def get_hydra_config(dir_path: Union[Path, str]) -> HydraConfig:
    pass


def configure_model(config) -> nn.Model:
    pass


def configure_logger(config) -> SummaryWriter:
    pass


def configure_task(config) -> TaskProtocol:
    pass


if __name__ == "__main__":
    mode, dir_path = main()

    print("All runs are finished. Starting test for the best checkpoint")

    if mode == "multirun":
        dir_path = find_best_subdir(dir_path)
    config = get_hydra_config(dir_path)
    model = configure_model(config)
    logger = configure_logger(config)
    task = configure_task(config)
    _, _, test_dl = task.build_dataloaders(config)
    runner = instantiate(
        config, logger=logger, output_dir=dir_path, model=model, task=task
    )
    runner.test(test_dl)
