# STL + Chronos Forecasting Pipeline

This repository contains an implementation and evaluation of a short-term load forecasting pipeline following the experiments from the preprint:

The Decomposition Penalty: Quantifying the Impact of Classical Pre-processing on Time-Series Foundation Models in Load Forecasting

DOI: https://doi.org/10.5281/zenodo.18828154
Zenodo: https://zenodo.org/doi/10.5281/zenodo.18828154

Authors: Nikhil Vinod
License: CC BY 4.0 (per Zenodo)

## Overview

This project implements the pipeline used in the study that measures the impact of classical STL decomposition on Time-Series Foundation Models (Chronos-T5) and compares them with classical baselines (XGBoost, SeasonalNaive, AutoETS).

Key components
- `data_loader.py` — load and preprocess `data/PJM_Load_hourly.csv`, create rolling evaluation windows
- `decomposition.py` — STL decomposition (period=168), trend projection, seasonal repetition, frozen Day-of-Week (DoW) residual adjustments computed from training data
- `models/` — Chronos LoRA wrapper, XGBoost baseline, StatsForecast baselines (SeasonalNaive + AutoETS)
- `train.py` — training orchestration (XGBoost training, Chronos pipeline loading, LoRA fine-tuning)
- `evaluate.py` — full evaluation pipeline, ablation study across 4 variants, exports `results.csv` and `ablation.csv`
- `visualize.py` — generate plots in `plots/` (ablation bar chart, error distributions, MAE/RMSE comparison, MAPE ranking)
- `utils.py` — helpers: seeding, device selection, metrics

The experiments in the paper run 107 rolling windows on the PJM hourly load dataset and evaluate MAE, RMSE, and MAPE.

## Quickstart

1. Install dependencies (recommended to use a virtualenv):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run a dry-run (fast, limited windows):

```bash
python evaluate.py --dry-run
```

3. Run full evaluation (may be slow and GPU-intensive):

```bash
python evaluate.py
```

4. Generate visualizations (after evaluation):

```bash
python visualize.py
```

Notes about GPU and training
- The pipeline detects CUDA and will run Chronos and LoRA training on GPU when available.
- LoRA fine-tuning implements mixed precision, gradient accumulation, and gradient checkpointing to stay within typical 6GB VRAM constraints.

## Reproducing the paper

- The experiments and code follow the preprint: "The Decomposition Penalty..." (see DOI above).
- The Zenodo record includes the preprint PDF `The_Decomposition_Penalty.pdf`.
- Citation (BibTeX / APA):

```
Vinod, N. (2026). The Decomposition Penalty: Quantifying the Impact of Classical Pre-processing on Time-Series Foundation Models in Load Forecasting. Zenodo. https://doi.org/10.5281/zenodo.18828154
```

## Files of interest
- `config.yaml` — dataset paths, model IDs, hyperparameters, dry-run controls
- `data/PJM_Load_hourly.csv` — hourly PJM load (required input)
- `results.csv`, `ablation.csv` — outputs from `evaluate.py`
- `plots/` — figure outputs

## Contact
Author: Nikhil Vinod — nikhilvinod321@gmail.com

---
Generated from the Zenodo preprint (DOI above) and the repository implementation.