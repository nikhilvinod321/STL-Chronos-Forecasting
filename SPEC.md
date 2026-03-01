# PROJECT SPECIFICATION: Short-Term Load Forecasting Pipeline
**Target Publication:** IEEE/Springer
**Core Application:** Peak Load Shaving & Grid Advisory Systems
**Core Objective:** Evaluate a hybrid STL + LoRA + Foundation Model pipeline against classical and ML baselines.

## 🤖 AI AGENT INSTRUCTIONS (READ FIRST)
* You are GitHub Copilot assisting a Lead ML Engineer.
* This document is the absolute source of truth. Assume `SPEC.md` and all files sit directly at the repository root.
* Do not redesign the project. Only modify or create files as instructed. Never use `# TODO` placeholders. Write executable code.
* **Implementation Order:** Implement files in this sequence: data_loader.py → decomposition.py → models/ (all) → train.py → evaluate.py → visualize.py. Validate each file works before proceeding to the next.

## 💻 HARDWARE CONSTRAINTS & VRAM MANAGEMENT
**Target GPU:** NVIDIA RTX 3060 (6GB VRAM)**
All PyTorch training loops MUST implement:
* Mixed precision (`torch.cuda.amp.autocast()`).
* Gradient accumulation. Per-device batch size = 4, accumulation steps = 4, effective batch size = 16.
* Memory clearing: `torch.cuda.empty_cache()` and `gc.collect()` after each epoch.
* Gradient Checkpointing: MUST enable `model.gradient_checkpointing_enable()` for the T5 model before applying LoRA.
* CPU Fallback: If `torch.cuda.is_available()` is False, run on CPU.

## 🤖 MODEL & DECOMPOSITION ARCHITECTURE
* **Foundation Model:** `amazon/chronos-t5-small`. Use `from chronos import ChronosPipeline`.
* **LoRA Configuration:** Rank=16, Alpha=32 (2× rank), Dropout=0.05, target_modules=["q", "k", "v", "o"].
* **LoRA Application:** Extract `pipeline.model` (T5ForConditionalGeneration), apply LoRA via PEFT, then use for fine-tuning.
* **Forecast Horizon:** 48 hours. **Lookback Window:** 336 hours.
* **Decomposition Pipeline (STL → DoW → Residual):**
  1. **Base Fit:** Fit `STL(period=168, robust=True)` on the 336h context window.
  2. **Trend:** Fit linear regression on the *last 48h* of the trend, project forward 48h.
  3. **Seasonality:** Repeat the last 168h block of the seasonal component for 48h.
  4. **Day of Week (DoW) [STRICT NO-LEAKAGE]:** Calculate the historical average of STL residuals per DoW using **ONLY the 2015-2017 training data**. Freeze these 7 values. Apply them to context/target windows via calendar lookup.
  5. **Residual:** Context = (Raw - Trend - Seasonality - DoW). Target = Predicted by Chronos (median/point forecast via `pipeline.predict()`).
  6. **Recomposition:** Pure pointwise addition: `Trend + Seasonality + DoW + Chronos_Residual`.

## 🧱 DATASET & EVALUATION PROTOCOL
* **Dataset:** PJM East hourly (`data/PJM_Load_hourly.csv`). Map columns to `Datetime` and `PJME_MW`.
* **Train/Test Split:** Train = 2015–2017. Test = 2018.
* **Rolling Window:** Stride 48h (strictly non-overlapping targets, overlapping context windows). 
* **Data Loader:** Must prevent future leakage and create `context -&gt; target` pairs.

## ⚙️ EXECUTION & CONFIGURATION
* **Reproducibility:** Fixed random seed (42) applied globally (`torch`, `numpy`, `random`, `xgboost`, `statsforecast`).
* **`--dry-run` Mode:** Limit test set to 2 rolling windows (96h). Max 1 epoch and 5 steps for Chronos LoRA. Limit `AutoETS` search space to `seasonal_periods=[168]` and `models=['AAA', 'AA']` only.

## 🔬 BASELINE MODELS
1. **Classical (`statsforecast` ONLY):**
   * Models: `SeasonalNaïve` and `AutoETS`.
   * **Logging:** Extract the exact ETS notation (e.g., `ETS(A,Ad,A)`) via `.model` or `.summary()` and append to `baseline_configs.txt`.
2. **Machine Learning (XGBoost):**
   * Use lags `[1, 2, ..., 24, 168]` from the context window.
   * **Multi-step strategy:** Wrap XGBoost in `sklearn.multioutput.MultiOutputRegressor` to predict the 48h horizon directly.

## 🚀 PROPOSED ABLATION STUDY
Evaluate these specific variants to isolate component value:
1. Raw Chronos (Zero-shot)
2. STL + Chronos (Zero-shot on residuals, NO DoW)
3. STL + DoW + Chronos (Zero-shot on residuals)
4. STL + DoW + Chronos + LoRA (HF PEFT with config above).

## 📏 METRICS & VISUALIZATION
* **Metrics:** Compute MAE, RMSE, and MAPE (add `1e-5` to denominator to prevent div-by-zero). Calculate mean and standard deviation across windows. Export to `results.csv` and `ablation.csv`.
* **Visualization (`visualize.py`):** Forecast vs actual plot, error distribution, and ablation bar chart.

## 📂 REQUIRED PROJECT STRUCTURE
```text
├── config.yaml
├── requirements.txt
├── data_loader.py
├── decomposition.py
├── models/
│   ├── __init__.py
│   ├── chronos_lora.py
│   ├── xgboost_baseline.py
│   └── statsforecast_baselines.py
├── train.py
├── evaluate.py
├── visualize.py
└── utils.py