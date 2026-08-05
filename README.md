<div align="center">

# SIER

### Systematic Indonesian Equity Research

**Foundation-model forecasting and systematic stock ranking for the Indonesian equity market.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Market](https://img.shields.io/badge/Market-IDX-E31E24?style=for-the-badge)](https://www.idx.co.id/)
[![Timeframes](https://img.shields.io/badge/Timeframes-1D_%7C_30m_%7C_15m-6C5CE7?style=for-the-badge)](#research-scope)

[![Universe](https://img.shields.io/badge/Equity_Universe-958-00A86B?style=flat-square)](#training-configuration)
[![Training](https://img.shields.io/badge/Training_Windows-800K-FF8C00?style=flat-square)](#training-configuration)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA_A100-76B900?style=flat-square&logo=nvidia&logoColor=white)](#training-configuration)
[![Tests](https://img.shields.io/badge/Tests-15%2F15_Passing-2EA44F?style=flat-square)](#reproducibility)

[Research Scope](#research-scope) · [Architecture](#system-architecture) · [Findings](#current-findings) · [Run Locally](#reproducibility) · [Limitations](#limitations-and-next-steps)

</div>

---

SIER is an end-to-end research pipeline for forecasting and ranking Indonesian equities with Python, PyTorch, Kronos, and `yfinance`.

SIER fine-tunes a time-series foundation model across 958 IDX-listed equities, produces short-horizon forecasts at daily and intraday resolutions, and converts those forecasts into cross-sectional stock rankings. The final selection layer combines model output with point-in-time price and volume confirmation for breakouts, reversals, market structure, momentum, and volume patterns.

> [!IMPORTANT]
> This repository is a research project, not investment advice. Historical and live-screening observations do not guarantee future performance.

## Project at a glance

| Universe | Primary model | Production timeframe | Training compute | Evaluation |
|:---:|:---:|:---:|:---:|:---:|
| **958 equities** | **Kronos** | **15 minutes** | **NVIDIA A100** | **Rolling backtests** |

| Research layer | Role |
|---|---|
| Forecasting | Generate probabilistic short-horizon price and volume paths |
| Cross-sectional ranking | Compare forecast strength across the full IDX universe |
| Technical confirmation | Validate breakout, reversal, structure, momentum, and volume setups |
| Risk filtering | Control for liquidity, volatility, abnormal activity, and market regime |

## Research scope

- Fine-tuning and inference for daily, 30-minute, and 15-minute OHLCV data.
- Systematic ranking using predicted returns, probability of an upward move, liquidity, volatility, and abnormal price-volume activity.
- Causal technical confirmation and penalty rules designed to reduce weak or hard-to-trade candidates.
- Leakage-controlled rolling backtests across multiple forecast horizons and market regimes.
- Reproducible local and Kaggle workflows, including GPU training and full-universe screening.

## System architecture

```text
IDX universe and OHLCV data
        |
        v
Kronos fine-tuning (daily / 30m / 15m)
        |
        v
Probabilistic short-horizon forecasts
        |
        v
Cross-sectional forecast ranking
        |
        v
Causal technical confirmation and penalties
        |
        v
Liquidity, volatility, and regime filters
        |
        v
Top candidates and rolling backtest evaluation
```

## Repository structure

| Directory | Purpose |
|---|---|
| `Kronos IDX FineTune/` | Daily model training, forecasting, checkpoints, and data updates |
| `Kronos IDX FineTune 15 Minutes/` | Intraday 15-minute fine-tuning pipeline and production checkpoint |
| `Kronos IDX FineTune 30 Minutes/` | Intraday 30-minute experiments |
| `Daily Screener/` | Current 15-minute next-session screening workflow |
| `BackTest Kronos Screener/` | Rolling forecast and ranking backtests |
| `BackTest Pattern Screener/` | Causal technical-pattern reranking and evaluation |
| `Model Screening UMA/` | Unusual Market Activity screening experiments |
| `Analisis Pendahuluan/` | Exploratory analysis of large IDX price moves |

Each directory contains its own README with workflow-specific instructions.

## Training configuration

The production intraday experiment was trained on an NVIDIA A100 using 200,000 sampled training windows per epoch and four selected predictor epochs, equivalent to 800,000 sampled windows across the run. This is the training-window count, not the model's attention context length. The 15-minute model uses a 240-bar lookback, a 20-bar forecast horizon, and `max_context=512`.

The model universe covers 958 Indonesian equities. Actual eligible counts can be lower at a given timestamp because of listing age, suspension, missing bars, or insufficient history.

## Current findings

### 1. The 15-minute timeframe is the most useful for short-horizon screening

Current live observations indicate that the 15-minute model is better than the tested higher timeframes at surfacing at least one stock that subsequently reaches or approaches Auto Rejection Atas (ARA). It does not yet identify the eventual ARA stock reliably as a single direct prediction, so forecast ranking still requires a technical selection layer.

Two early forward observations illustrate the behavior:

| Screen | Candidate | Model placement | Observed move |
|---|---|---:|---:|
| 3 August 2026 | **GTSI** | TF15 top 5 | **≥15%** |
| 4 August 2026 | **CBPE** | TF15 rank 7 | **25% ARA** |

These are live case studies, not a statistically sufficient performance claim. More forward observations are required before estimating hit rates or expected returns.

### 2. Fine-tuning quality depends on context, strategy, and data hygiene

Model adaptation alone is insufficient. The training context and prediction target must match the intended trading decision, while the current dataset still contains incomplete bars, suspensions, zero-volume periods, corporate-action effects, and heterogeneous issuer behavior. Domain knowledge should therefore inform universe construction, issuer-level exclusions, liquidity constraints, and regime-specific calibration.

### 3. Forecasts work best as a candidate generator

The strongest current use of the foundation model is to compress a broad IDX universe into a manageable candidate set. Causal technical confirmation then improves prioritization among those candidates. In the existing 41-origin daily pattern experiment, reranking improved top-5 precision from 49.27% to 54.15% for the validated checkpoint and from 52.68% to 55.61% for the production checkpoint. These figures are full-timeframe optimization results, not untouched out-of-sample estimates.

| Checkpoint | Base top-5 precision | With technical reranking | Change |
|---|---:|---:|---:|
| Validated no-refit E15 | 49.27% | **54.15%** | **+4.88 pp** |
| Production refit E4 | 52.68% | **55.61%** | **+2.93 pp** |

## Reproducibility

Clone the repository with its Kronos submodule and enable Git LFS:

```bash
git clone --recurse-submodules https://github.com/zzeiidann/SIER.git
cd SIER
git lfs install
git lfs pull
```

For the current local screening workflow:

```bash
cd "Daily Screener"
uv sync
uv run python update_tf15_parquet.py
uv run python project_tf15_next_session.py
```

For causal pattern backtests:

```bash
python3 -m pip install -r "BackTest Pattern Screener/requirements.txt"
python3 "BackTest Pattern Screener/run_pattern_backtest.py"
python3 -m unittest discover -s "BackTest Pattern Screener/tests" -v
```

## Methodological safeguards

- Rolling technical levels are shifted so the evaluated candle cannot define its own support or resistance.
- Future returns and maximum favorable/adverse excursions are separated from live feature generation.
- Whole origins with inadequate forecast coverage are excluded before ranking optimization.
- Results distinguish validated frozen checkpoints from production refit checkpoints.
- In-sample ranking optimization is labeled explicitly and is not presented as out-of-sample evidence.

## Limitations and next steps

- Clean and version data snapshots, including corporate actions, delistings, suspensions, and zero-volume bars.
- Add point-in-time issuer metadata and sector-aware filters.
- Expand walk-forward evaluation with untouched chronological holdouts.
- Calibrate ranking and technical thresholds separately by liquidity and market regime.
- Accumulate a larger forward-test sample for the 15-minute workflow before drawing performance conclusions.

## Technology

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/Parquet-50ABF1?style=flat-square&logo=apacheparquet&logoColor=white" alt="Apache Parquet">
  <img src="https://img.shields.io/badge/Git_LFS-F05032?style=flat-square&logo=git&logoColor=white" alt="Git LFS">
</p>

Additional components include `yfinance`, Optuna, Git LFS, and the Kronos time-series foundation model.

## Acknowledgements

SIER builds on the open-source [Kronos](https://github.com/shiyu-coder/Kronos) time-series foundation model. The upstream source is included as a Git submodule and retains its original license and attribution.
