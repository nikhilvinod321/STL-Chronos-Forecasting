# STL + Chronos Forecasting Pipeline

This repository contains the implementation and evaluation of a short-term load forecasting pipeline accompanying the preprint:

**The Decomposition Penalty: Quantifying the Impact of Classical Pre-processing on Time-Series Foundation Models in Load Forecasting**

DOI: https://doi.org/10.5281/zenodo.18828154  
Author: Nikhil Vinod  
License: CC BY 4.0 (per Zenodo)

---

## Overview

This project measures whether classical STL decomposition helps or hurts
Time-Series Foundation Models (Chronos-T5, Chronos-Bolt-Small, Lag-Llama) versus
classical baselines across **three geographically and temporally diverse datasets**:

| Dataset | Description | Frequency | Eval windows |
|---|---|---|---|
| **PJM** | Hourly US electricity load (2015–2018) | 60 min | 107 |
| **ETTm1** | 15-min transformer temperature, China | 15 min | 580 |
| **UCI Electricity** | 321-client Portuguese hourly consumption (2012–2014) | 60 min | varies |

The pipeline evaluates **9 model variants** per dataset:

| Variant | Description |
|---|---|
| SeasonalNaive | Repeat last seasonal period |
| AutoETS | Automatic exponential smoothing (StatsForecast) |
| XGBoost | Gradient boosting with calendar + lag features |
| Chronos (raw) | Amazon Chronos-T5 zero-shot |
| STL + Chronos | STL residual fed to Chronos |
| STL + DoW + Chronos | STL + day-of-week projection |
| STL + DoW + Chronos + LoRA | Above + 10-epoch LoRA fine-tuning |
| Lag-Llama (raw) | Non-Amazon decoder-only foundation model (zero-shot) |
| STL + DoW + Lag-Llama | STL + DoW residual fed to Lag-Llama |

---

## Key Results

### Cross-dataset summary

| Model | PJM MAPE ↓ | PJM MAE | ETTm1 sMAPE ↓ | ETTm1 MAE ↓ | UCI MAPE ↓ | UCI MAE |
|---|---|---|---|---|---|---|
| SeasonalNaive | 11.11% | 3622 | 33.78% | 1.712 | 7.13% | 43141 |
| AutoETS | 13.12% | 4096 | **22.22%** | **0.998** | 42.76% | 201343 |
| XGBoost | **7.69%** | **2486** | 25.93% | 1.300 | **5.63%** | **26328** |
| Chronos (raw) | 9.21% | 2980 | 26.69% | 1.339 | 6.28% | 32271 |
| STL + Chronos | 12.86% | 4150 | 38.85% | 1.905 | 8.47% | 50093 |
| STL + DoW + Chronos | 13.43% | 4316 | 38.87% | 1.911 | 8.54% | 50427 |
| STL + DoW + Chronos + LoRA | 13.44% | 4322 | 38.97% | 1.914 | 8.56% | 50709 |
| Lag-Llama (raw) | 24.36% | 7478 | 33.10% | 1.969 | 25.65% | 131100 |
| STL + DoW + Lag-Llama | 12.79% | 4129 | 39.72% | 1.913 | 8.41% | 49789 |

> ETTm1 MAPE is unreliable (near-zero OT values); sMAPE and MAE are the primary ETTm1 metrics.

### Key findings

1. **STL preprocessing consistently hurts** all foundation models (Chronos and Lag-Llama) on all three datasets. Raw zero-shot inference is strictly better.
2. **LoRA fine-tuning on decomposed residuals provides negligible benefit** — only +0.58 pp MAPE degradation on PJM, indicating the STL bottleneck dominates.
3. **XGBoost wins on electricity load** (PJM 7.69%, UCI 5.63%) despite being a shallow model; AutoETS wins on ETTm1.
4. **Lag-Llama (raw) performs poorly** on all datasets (24–26% MAPE on electricity), but STL+DoW rescues it to near-Chronos levels — consistent with the decomposition penalty benefiting weaker zero-shot models.
5. **Attention entropy** is identical (6.32 bits) for raw vs STL-decomposed inputs — Chronos tokenisation normalises both to similar discrete representations.
6. Findings are **robust across geographies and sampling rates**: US grid (60 min), Chinese transformer (15 min), Portuguese household electricity (60 min).

---

## Repository Structure

```
stl-chronos-forecasting/
├── data/
│   ├── ETTm1.csv                        # ETTm1 dataset
│   ├── PJM_Load_hourly.csv              # PJM hourly load
│   ├── UCI_electricity.csv              # UCI Electricity (321 clients)
│   └── electricity_hourly_dataset.tsf  # MonashTSF format (used by data_loader)
├── models/
│   ├── chronos_lora.py                  # Chronos + LoRA fine-tuning wrapper
│   ├── foundation_models.py             # Chronos-Bolt + Lag-Llama wrappers
│   ├── statsforecast_baselines.py       # SeasonalNaive, AutoETS via StatsForecast
│   └── xgboost_baseline.py             # XGBoost with calendar features
├── plots/                               # Generated charts (gitignored large files)
├── config.yaml                          # Datasets, model, eval settings
├── data_loader.py                       # Dataset loaders + rolling-window splitter
├── decomposition.py                     # STL, DoW projection, residual reconstruction
├── evaluate.py                          # Main evaluation loop → results.csv, ablation.csv
├── make_table.py                        # Console table + LaTeX export
├── train.py                             # DATASET_META, training helpers
├── utils.py                             # Seeding, device selection, metrics
├── visualize.py                         # All plots (per-dataset + cross-dataset)
├── lora_analysis.py                     # LoRA rank sweep + learning curves
├── attention_viz.py                     # Encoder attention entropy visualisation
├── results.csv                          # Full per-window results
├── ablation.csv                         # Per-model aggregated summary
└── requirements.txt
```

---

## Quickstart

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Install Lag-Llama (no-deps to avoid gluonts version conflicts)
pip install --no-deps git+https://github.com/time-series-foundation-models/lag-llama.git

python evaluate.py --dry-run        # fast 2-window sanity check (all 3 datasets)
python evaluate.py                  # full evaluation (GPU strongly recommended)
python visualize.py                 # generate all 10 plots
python lora_analysis.py --mode curves  # LoRA learning curves
python attention_viz.py             # attention heatmaps
python make_table.py                # print results table + export LaTeX
```

### Lag-Llama checkpoint

Lag-Llama is automatically downloaded via HuggingFace on first use:

```python
# Happens automatically inside models/foundation_models.py:
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download("time-series-foundation-models/Lag-Llama", "lag-llama.ckpt")
```

---

## GPU Setup

Set `device: "auto"` in `config.yaml` (default). With `torch.cuda.is_available()` returning True,
all foundation models run on GPU automatically.

If CUDA is not detected despite having a GPU:

```bash
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

LoRA fine-tuning: ~24 s/epoch on RTX 3060 vs ~106 s on CPU.

> **PyTorch 2.6 note**: The pipeline uses `torch.serialization.add_safe_globals` to
> safely unpickle the Lag-Llama checkpoint under PyTorch 2.6's strict `weights_only` policy.

---

## Configuration (`config.yaml`)

```yaml
datasets: ["PJM", "ETTm1", "UCI_Electricity"]
second_foundation_model: "lag-llama"   # or "chronos-bolt"
device: "auto"
horizon: 24
context_multiplier: 4
lora_rank: 8
lora_epochs: 10
```

---

## Metric Notes

| Dataset | Primary metric | Notes |
|---|---|---|
| PJM | MAPE (%) | Load values always >> 0 |
| UCI Electricity | MAPE (%) | Household consumption always >> 0 |
| ETTm1 | sMAPE (%) + MAE | OT (Oil Temp) can be near-zero; MAPE unreliable |

sMAPE is bounded 0–200% and computed by `utils.compute_metrics`.

---

## Output Files

| File | Description |
|---|---|
| `results.csv` | Per-window MAE/RMSE/MAPE/sMAPE for all datasets × models |
| `ablation.csv` | Per-model aggregated summary (mean ± std) |
| `plots/ablation_bar_<DS>.png` | Ablation bar chart per dataset |
| `plots/error_distribution_<DS>.png` | Box plots per dataset |
| `plots/ranking_<DS>.png` | Horizontal rank chart per dataset |
| `plots/cross_dataset_comparison.png` | Normalised MAE across all 3 datasets |
| `plots/lora_history_<DS>.json` | LoRA train/val loss history per dataset |
| `plots/lora_sweep_results.json` | LoRA rank sweep results |
| `plots/attention_entropy_multi.json` | Attention entropy data |

---

## Reproducing the Paper

- The Zenodo record includes `The_Decomposition_Penalty.pdf`.
- Full evaluation: `python evaluate.py` covers all 3 datasets and 9 model variants.
- Expected runtime: ~2–4 hours on GPU (RTX 3060), ~10–15 hours on CPU.

---

## Citation

```bibtex
@misc{vinod2026decomposition,
  author    = {Vinod, Nikhil},
  title     = {The Decomposition Penalty: Quantifying the Impact of Classical
               Pre-processing on Time-Series Foundation Models in Load Forecasting},
  year      = {2026},
  doi       = {10.5281/zenodo.18828154},
  url       = {https://doi.org/10.5281/zenodo.18828154}
}
```

## Contact

Author: Nikhil Vinod — nikhilvinod321@gmail.com
