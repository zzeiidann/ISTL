# SIER

**Systematic Indonesian Equity Research** is an end-to-end research pipeline for forecasting and ranking Indonesian equities with Python, PyTorch, Kronos, and `yfinance`.

SIER fine-tunes a time-series foundation model across 958 IDX-listed equities, produces short-horizon forecasts at daily and intraday resolutions, and converts those forecasts into cross-sectional stock rankings. The final selection layer combines model output with point-in-time price and volume confirmation for breakouts, reversals, market structure, momentum, and volume patterns.

> This repository is a research project, not investment advice. Historical and live-screening observations do not guarantee future performance.

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

- The 3 August 2026 screen ranked GTSI in the 15-minute top five for 4 August; GTSI subsequently advanced by at least 15%.
- The 4 August 2026 screen ranked CBPE seventh for 5 August; CBPE subsequently reached a 25% ARA move.

These are live case studies, not a statistically sufficient performance claim. More forward observations are required before estimating hit rates or expected returns.

### 2. Fine-tuning quality depends on context, strategy, and data hygiene

Model adaptation alone is insufficient. The training context and prediction target must match the intended trading decision, while the current dataset still contains incomplete bars, suspensions, zero-volume periods, corporate-action effects, and heterogeneous issuer behavior. Domain knowledge should therefore inform universe construction, issuer-level exclusions, liquidity constraints, and regime-specific calibration.

### 3. Forecasts work best as a candidate generator

The strongest current use of the foundation model is to compress a broad IDX universe into a manageable candidate set. Causal technical confirmation then improves prioritization among those candidates. In the existing 41-origin daily pattern experiment, reranking improved top-5 precision from 49.27% to 54.15% for the validated checkpoint and from 52.68% to 55.61% for the production checkpoint. These figures are full-timeframe optimization results, not untouched out-of-sample estimates.

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

Python, PyTorch, pandas, NumPy, `yfinance`, Hugging Face, Optuna, Parquet, Jupyter, Kaggle, Git LFS, and the Kronos time-series foundation model.

## Acknowledgements

SIER builds on the open-source [Kronos](https://github.com/shiyu-coder/Kronos) time-series foundation model. The upstream source is included as a Git submodule and retains its original license and attribution.
