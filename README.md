# Repo

## Intro

This repo is my pet project with utility functions and general functional used in most time-series researches. I'm going to transform it into a template for my future repos.

## Quickstart

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run.py train.device=cpu
python run.py sanity_run=False train.device=cpu train.epochs=1
```

The default config runs a forecasting sanity check on the ETT dataset. Put the
CSV files under `data/ETT/` with names such as `ETTh1.csv`, or override
`data=...` with another dataset config.

Useful model overrides:

```bash
python run.py sanity_run=False train.device=cpu train.epochs=1 models=forecasting_cnn
python run.py sanity_run=False train.device=cpu train.epochs=1 models=mlp
python run.py sanity_run=False train.device=cpu train.epochs=1 models=naive_forecast
python run.py sanity_run=False train.device=cpu train.epochs=1 models=nontrainable_ewma
```

Set `train.device=cuda` only when CUDA is available. If you omit it, the runner
falls back to CPU when CUDA cannot be used.

To add a new model to this pipeline, you need to perform the following steps:

1. Create model file in ./sandbox/models/. The key class must implement methods
   forward, predict, configure_optimizers, save_checkpoint and load_checkpoint (watch the contract in this file or just use ./sandbox/models/mlp.py for reference)
   The model have to extract the input tensor/tensors from the batch by itself. The structure of a batch is written at the end of this file.
   The model should receive specific arguments in the constructor, not just config.
2. If the model requires specific losses, implement them.
3. Create model, optimizer, dataset and loss (if needed) configs in.

The basic pipeline follows the contract below.
## Tasks

The repository is versatile enough to perform the following tasks:

- forecasting [to do]:
  - autoregressive [to do]
  - non-autoregressive [to do]
- reconstruction [to do]

- anomaly detection (score-based, window-wise task statement) [to do]
  - reconstruction-based
  - prediction-based
  - mixed.

# Contracts for sandbox and test functions

## Task

- `build_dataloaders(cfg) -> Tuple[nn.DataLoader](train, val, test)`

- `loss_fn(outputs, batch) -> loss (torch.Tensor?)`

- `metrics(pred, batch) -> dict[str, float]`

- `visualize(pred, batch, outdir) -> None`

## Model

- `forward(batch*: torch.Tensor) -> Union[torch.Tensor, Tuple(torch.Tensor)]`.
    Here we don't expect all the models to follow this type hinting,
    we just use is for convenience.
- `predict(batch*: torch.Tensor) -> Union[torch.Tensor, Tuple(torch.Tensor)]`.
    Is a wrapped version of `forward` used for convenience and code readability.

- `configure_optimizers(cfg) -> (optimizer, scheduler|None)`.
    The model does not necessarily have to have this method.
    Personally I usually configure optimizers in a separate train.py file.

## Runner

- `train(task, dataloaders, cfg) -> artifacts (checkpoints, test results)`

- `eval(task, dataloader, cfg) -> metrics + optional preds with pictures`

- `test(task, dataloader, cfg) -> metrics + optional preds with pictures`

## SlidingWindowDataset

- `__getitem__(self, idx) -> Union[Tuple[torch.Tensor], Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]], torch.Tensor]`.
The output of  `__getitem__` function can be either one Tensor in case of simple reconstruction task, Tuple[torch.Tensor] in case of simple forecasting task without predictors, and Tuple of Tuples (Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]) for transformers and other models, for instance.

## Batch format

- `batch = { "x": (B,T,F_in), "y": (B,H,F_out), "x_t":  (B, T, TF) | None,
"y_t": Optional (B, H, TF) | None, "meta": {...} }`.  I'm not really sure what I shoud do in this case. In case when the task == reconstruction or `ad_recontruction`

The main idea behind batch format is as follows: everything that can be potentially used in model forward during training and inference lays on the first layer. The other data, e.g. used in debugging, is hidden in "meta".
