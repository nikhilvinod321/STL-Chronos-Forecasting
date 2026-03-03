import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

"""
Generate paper-quality results tables (console + LaTeX) from ablation.csv.
Handles 2 or 3 datasets (PJM, ETTm1, UCI_Electricity) and all model variants
including Lag-Llama.  Columns present in ablation.csv are discovered at runtime
so the script stays forward-compatible.
"""
import pandas as pd
import numpy as np
import math

df = pd.read_csv("ablation.csv")

ALIAS = {
    "SeasonalNaive":                              "SeasonalNaive",
    "AutoETS":                                    "AutoETS",
    "XGBoost":                                    "XGBoost",
    "Raw Chronos (Zero-shot)":                    "Chronos (raw)",
    "STL + Chronos (Zero-shot)":                  "STL + Chronos",
    "STL + DoW + Chronos (Zero-shot)":            "STL + DoW + Chronos",
    "STL + DoW + Chronos + LoRA":                 "STL + DoW + Chronos + LoRA",
    "Raw CHRONOS-BOLT-SMALL (Zero-shot)":         "Chronos-Bolt (raw)",
    "STL + DoW + CHRONOS-BOLT-SMALL (Zero-shot)": "STL + DoW + Bolt",
    # Lag-Llama (non-Amazon decoder-only model)
    "Raw LAG-LLAMA (Zero-shot)":                  "Lag-Llama (raw)",
    "STL + DoW + LAG-LLAMA (Zero-shot)":          "STL + DoW + Lag-Llama",
}

def _get(ds_df, model_full, col):
    """Safely retrieve a value; return NaN if model or column is absent."""
    if model_full not in ds_df.index or col not in ds_df.columns:
        return float("nan")
    val = ds_df.loc[model_full, col]
    return float(val) if not isinstance(val, float) else val

def _fmt(val, fmt=".2f", fallback="  --  "):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return fallback
    return format(val, fmt)

# Build per-dataset sub-tables
datasets_present = df["Dataset"].unique().tolist()
ds_tables = {ds: df[df["Dataset"] == ds].set_index("Model") for ds in datasets_present}

pjm   = ds_tables.get("PJM",            pd.DataFrame())
ettm1 = ds_tables.get("ETTm1",          pd.DataFrame())
uci   = ds_tables.get("UCI_Electricity", pd.DataFrame())

# -- Console table --------------------------------------------------------------
W = 100
print("=" * W)
print("PAPER RESULTS TABLE")
print("PJM / UCI_Electricity : primary metric = MAPE (%)  |  lower is better")
print("ETTm1                 : primary metric = sMAPE (%) |  lower is better  (MAPE invalid: near-zero OT)")
print("=" * W)

has_uci = not uci.empty
head = (f"  {'Model':<34}  {'PJM MAPE':>9}  {'ETTm1 sMAPE':>12}"
        + (f"  {'UCI MAPE':>9}" if has_uci else "")
        + f"  {'PJM MAE':>9}  {'ETTm1 MAE':>10}")
print(head)
print("  " + "-" * (W - 2))

pjm_mape_min   = pjm["MAPE_mean"].min()   if not pjm.empty   else float("nan")
e1_smape_min   = ettm1["sMAPE_mean"].min() if (not ettm1.empty and "sMAPE_mean" in ettm1.columns) \
                 else float("nan")
uci_mape_min   = uci["MAPE_mean"].min()   if not uci.empty    else float("nan")
e1_mae_min     = ettm1["MAE_mean"].min()   if not ettm1.empty  else float("nan")

for m_full, m_short in ALIAS.items():
    pjm_mape  = _get(pjm,   m_full, "MAPE_mean")
    e1_smape  = _get(ettm1, m_full, "sMAPE_mean")
    uci_mape  = _get(uci,   m_full, "MAPE_mean")
    pjm_mae   = _get(pjm,   m_full, "MAE_mean")
    e1_mae    = _get(ettm1, m_full, "MAE_mean")

    # Skip rows where the model was not run at all
    if all(math.isnan(v) for v in [pjm_mape, e1_smape, uci_mape, pjm_mae, e1_mae]):
        continue

    bp  = " *" if pjm_mape  == pjm_mape_min  else "  "
    be  = " *" if e1_smape  == e1_smape_min  else "  "
    bu  = " *" if uci_mape  == uci_mape_min  else "  "
    bem = " *" if e1_mae    == e1_mae_min    else "  "

    row = (f"  {m_short:<34} "
           f"{_fmt(pjm_mape):>8}%{bp} "
           f"{_fmt(e1_smape):>11}%{be}"
           + (f" {_fmt(uci_mape):>8}%{bu}" if has_uci else "")
           + f" {_fmt(pjm_mae, '.1f'):>9}  "
             f"{_fmt(e1_mae, '.3f'):>10}{bem}")
    print(row)

print()
print(f"  (* best in column)")
if not pjm.empty:
    best_pjm = ALIAS.get(pjm["MAPE_mean"].idxmin(), pjm["MAPE_mean"].idxmin())
    print(f"  Best PJM MAPE:          {best_pjm}  ->  {pjm_mape_min:.2f}%")
if not ettm1.empty and "sMAPE_mean" in ettm1.columns:
    best_e1 = ALIAS.get(ettm1["sMAPE_mean"].idxmin(), ettm1["sMAPE_mean"].idxmin())
    print(f"  Best ETTm1 sMAPE:       {best_e1}  ->  {e1_smape_min:.2f}%")
if has_uci:
    best_uci = ALIAS.get(uci["MAPE_mean"].idxmin(), uci["MAPE_mean"].idxmin())
    print(f"  Best UCI_Elec MAPE:     {best_uci}  ->  {uci_mape_min:.2f}%")

# -- Key findings ---------------------------------------------------------------
print()
print("-" * W)
print("KEY FINDINGS")
print("-" * W)

if not pjm.empty:
    try:
        raw_c  = _get(pjm, "Raw Chronos (Zero-shot)",        "MAPE_mean")
        stl_c  = _get(pjm, "STL + Chronos (Zero-shot)",      "MAPE_mean")
        lora_c = _get(pjm, "STL + DoW + Chronos + LoRA",     "MAPE_mean")
        raw_b  = _get(pjm, "Raw CHRONOS-BOLT-SMALL (Zero-shot)",         "MAPE_mean")
        raw_ll = _get(pjm, "Raw LAG-LLAMA (Zero-shot)",       "MAPE_mean")
        print(f"  PJM -- STL+Chronos vs Raw Chronos:   {stl_c:.2f}% vs {raw_c:.2f}%  "
              f"({'worse' if stl_c > raw_c else 'better'} by {abs(stl_c-raw_c):.2f}pp)")
        print(f"  PJM -- LoRA delta (vs STL+DoW+Chronos): {lora_c:.2f}% vs {stl_c:.2f}%  "
              f"(delta = {lora_c - stl_c:+.3f}pp)")
        if not math.isnan(raw_ll):
            print(f"  PJM -- Lag-Llama (raw) MAPE: {raw_ll:.2f}%  "
                  f"vs Chronos (raw): {raw_c:.2f}%")
    except Exception:
        pass

if not ettm1.empty:
    try:
        e1_raw_c = _get(ettm1, "Raw Chronos (Zero-shot)",   "MAE_mean")
        e1_stl_c = _get(ettm1, "STL + Chronos (Zero-shot)", "MAE_mean")
        e1_raw_ll = _get(ettm1, "Raw LAG-LLAMA (Zero-shot)", "MAE_mean")
        print(f"  ETTm1 -- STL+Chronos vs Raw Chronos: {e1_stl_c:.3f} vs {e1_raw_c:.3f}  "
              f"({'worse' if e1_stl_c > e1_raw_c else 'better'})")
        if not math.isnan(e1_raw_ll):
            print(f"  ETTm1 -- Lag-Llama (raw) MAE: {e1_raw_ll:.4f}  "
                  f"vs Chronos (raw): {e1_raw_c:.4f}")
    except Exception:
        pass

if has_uci:
    try:
        u_raw_c  = _get(uci, "Raw Chronos (Zero-shot)",   "MAPE_mean")
        u_stl_c  = _get(uci, "STL + Chronos (Zero-shot)", "MAPE_mean")
        u_raw_ll = _get(uci, "Raw LAG-LLAMA (Zero-shot)",  "MAPE_mean")
        print(f"  UCI -- STL+Chronos vs Raw Chronos:   {u_stl_c:.2f}% vs {u_raw_c:.2f}%  "
              f"({'worse' if u_stl_c > u_raw_c else 'better'} by {abs(u_stl_c-u_raw_c):.2f}pp)")
        if not math.isnan(u_raw_ll):
            print(f"  UCI -- Lag-Llama (raw) MAPE: {u_raw_ll:.2f}%  "
                  f"vs Chronos (raw): {u_raw_c:.2f}%")
    except Exception:
        pass

# -- LaTeX table ----------------------------------------------------------------
print()
print("-" * W)
print("LaTeX:")
print("-" * W)

n_ds_cols = 2 * len(datasets_present)   # 2 metrics per dataset
col_spec   = "l" + "".join("|cc" for _ in datasets_present)

print(r"\begin{table}[t]")
print(r"\centering")
caption = (r"Forecasting Results. $\downarrow$ = lower is better. "
           r"PJM and UCI-Electricity: MAPE~(\%). "
           r"ETTm1: sMAPE~(\%) (MAPE unreliable due to near-zero Oil Temperature). "
           r"Lag-Llama is a non-Amazon decoder-only foundation model included for "
           r"cross-architecture comparison. "
           r"Bold = best per column.")
print(r"\caption{" + caption + r"}")
print(r"\label{tab:results}")
print(r"\begin{tabular}{" + col_spec + r"}")
print(r"\toprule")

# Header row 1: dataset names
hdr1 = r"\multirow{2}{*}{\textbf{Model}}"
ds_display = {"PJM": "PJM", "ETTm1": "ETTm1",
              "UCI_Electricity": r"UCI-Electricity"}
for i, ds in enumerate(datasets_present):
    sep = "|" if i < len(datasets_present) - 1 else ""
    hdr1 += (r" & \multicolumn{2}{" + sep + r"c|}{\textbf{" + ds_display.get(ds, ds) + r"}}")
print(hdr1 + r" \\")

# Header row 2: metric names
hdr2 = ""
for ds in datasets_present:
    if ds == "ETTm1":
        hdr2 += r" & sMAPE(\%)$\downarrow$ & MAE$\downarrow$"
    else:
        hdr2 += r" & MAPE(\%)$\downarrow$ & MAE$\downarrow$"
print(hdr2 + r" \\")
print(r"\midrule")

def _bold_if(val, best_val, fmt):
    s = _fmt(val, fmt, "--")
    if not math.isnan(val) and val == best_val:
        return r"\textbf{" + s + r"}"
    return s

# Pre-compute best values per dataset
bests = {}
for ds in datasets_present:
    tbl = ds_tables[ds]
    if ds == "ETTm1":
        bests[ds] = (
            tbl["sMAPE_mean"].min() if "sMAPE_mean" in tbl.columns else float("nan"),
            tbl["MAE_mean"].min()   if "MAE_mean"   in tbl.columns else float("nan"),
        )
    else:
        bests[ds] = (
            tbl["MAPE_mean"].min() if "MAPE_mean" in tbl.columns else float("nan"),
            tbl["MAE_mean"].min()  if "MAE_mean"  in tbl.columns else float("nan"),
        )

for m_full, m_short in ALIAS.items():
    cells = []
    any_data = False
    for ds in datasets_present:
        tbl = ds_tables[ds]
        if ds == "ETTm1":
            v1 = _get(tbl, m_full, "sMAPE_mean")
        else:
            v1 = _get(tbl, m_full, "MAPE_mean")
        v2 = _get(tbl, m_full, "MAE_mean")
        if not (math.isnan(v1) and math.isnan(v2)):
            any_data = True
        b1, b2 = bests[ds]
        mae_fmt = ".3f" if ds == "ETTm1" else ".0f"
        cells.append(_bold_if(v1, b1, ".2f") + " & " + _bold_if(v2, b2, mae_fmt))
    if not any_data:
        continue
    row_str = "  " + m_short + " & " + " & ".join(cells) + r" \\"
    print(row_str)

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
