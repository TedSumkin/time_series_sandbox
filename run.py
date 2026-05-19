"""Project entrypoint for Hydra-driven experiments."""

from __future__ import annotations

import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_experiment_header(cfg: DictConfig, out_dir: Path) -> None:
    """Print a compact experiment summary for terminal visibility."""
    experiment_name = cfg.get("experiment_name", "unnamed_experiment")
    seed = cfg.get("random_seed", {}).get("seed", "unset")
    print(f"Experiment: {experiment_name}")
    print(f"Output dir: {out_dir}")
    print(f"Seed: {seed}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> str:
    """Run a single train/test experiment from the Hydra configuration."""
    hydra_config = HydraConfig.get()
    out_dir = Path(hydra_config.runtime.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    log_experiment_header(cfg, out_dir)

    seed = int(cfg.get("random_seed", {}).get("seed", 42))
    set_global_seed(seed)

    logger = SummaryWriter(log_dir=str(out_dir / "tb"))
    try:
        task = instantiate(cfg.tasks)
        model = instantiate(cfg.models)
        runner = instantiate(
            cfg.runner,
            cfg,
            logger=logger,
            output_dir=out_dir,
            task=task,
            model=model,
        )

        # TO DO: move this block to runner code.
        # it looks unsafe
        train_dl, val_dl, test_dl = task.build_dataloaders(cfg)
        runner.train_dl = train_dl
        runner.val_dl = val_dl
        runner.test_dl = test_dl

        if cfg.get("sanity_run", False):
            print("Sanity run enabled - skipping training and testing.")
            print("Performing sanity checks on model and dataloaders...")
            check_results = runner.run_checks()
            print(f"Sanity check results: {check_results}")
            return str(out_dir)

        best_val_metrics = runner.train(train_dl=train_dl, val_dl=val_dl)

        runner.load_best_checkpoint()
        test_metrics = runner.test(test_dl=test_dl)

        summary_path = out_dir / "run_summary.txt"
        summary_lines = [
            f"experiment_name: {cfg.get('experiment_name', 'unnamed_experiment')}",
            f"seed: {seed}",
            f"best_val_metrics: {best_val_metrics}",
            f"test_metrics: {test_metrics}",
        ]
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    finally:
        logger.close()

    return str(out_dir)


if __name__ == "__main__":
    main()
