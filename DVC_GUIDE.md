# DVC Integration Guide

This repository is set up with a minimal DVC pipeline to reproduce training and version model artifacts.

## Prerequisites
- Install dev requirements (includes DVC):

```bash
pip install -r requirements_dev.txt
```

## Initialize DVC
Run once per repo to create the `.dvc` directory:

```bash
dvc init
```

## Pipeline
The pipeline has a single `train` stage defined in `dvc.yaml`:

- cmd: `python train.py`
- deps: `train.py`, `model.py`, `dataset.py`, `config.py`
- outs:
  - `models/runs/en_hi_model`
  - `tokenizer_en.json`
  - `tokenizer_hi.json`

Note: Logs under `runs/` (TensorBoard) and `mlruns/` (MLflow) are ignored by DVC via `.dvcignore`. You can add them as outs later if needed, but they can be large.

## Typical Workflow

```bash
# 1) Initialize DVC (first time only)
dvc init

# 2) Run pipeline and track outputs
dvc repro

# 3) Check status of pipeline
dvc status

# 4) Optionally push outs to a remote (e.g., S3)
dvc remote add -d s3 s3://your-bucket/transformer
# Configure credentials via environment or AWS config

dvc push
```

## Notes
- The dataset is fetched via Hugging Face (`cfilt/iitb-english-hindi`) during training; if you want DVC to manage the raw dataset, consider adding a separate stage that downloads data into a local folder and switch `train.py` to read from that path.
- To track hyperparameters with DVC `params`, you can create a `params.yaml` and adapt `train.py` to read from it.
