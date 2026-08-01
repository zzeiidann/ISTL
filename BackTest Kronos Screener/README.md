# BackTest Kronos Screener

Six standalone Kaggle notebooks compare three frozen Kronos checkpoints on two
rolling daily backtest modes:

| Checkpoint | 1-day | 5-day |
|---|---|---|
| Validated no-refit epoch 15 | `backtest_1d_validated_no_refit_e15.ipynb` | `backtest_5d_validated_no_refit_e15.ipynb` |
| Refit-run validated epoch 4 | `backtest_1d_refit_run_validated_e4.ipynb` | `backtest_5d_refit_run_validated_e4.ipynb` |
| Production refit epoch 4 | `backtest_1d_production_refit_e4.ipynb` | `backtest_5d_production_refit_e4.ipynb` |

`screen_2026_07_31_backtest_weights.ipynb` menjalankan screening live untuk
31 Juli 2026 dengan checkpoint validated no-refit E15 dan production refit E4.
Masing-masing model menggunakan bobot terbaik dari arsip backtest 1D miliknya,
menampilkan arti serta arah setiap bobot, menghasilkan top 30 per model, lalu
membuat top 30 consensus.

Each notebook clones `https://github.com/zzeiidann/ISTL.git`, pulls only its own
Git LFS checkpoint, installs PyTorch 2.3.1 CUDA 11.8 for Kaggle P100/T4, runs
frozen-model rolling inference, and performs 1,500 Optuna TPE trials. The runner
also clones the official Kronos source at the pinned tested commit when it is
not present in ISTL.

For every origin and horizon, the candidate universe is at most 100 stocks with
positive predicted daily close gain. A single global weighted percentile-rank
score selects 30. A realized hit requires:

```text
actual target-day high / actual previous-session close - 1 >= 5%
```

The engine uses 42 backtest origins, a chronological 70/30 optimization/holdout
split, and an embargo equal to the maximum forecast horizon. Outputs include
the full candidate panel, selected daily top 30, Optuna trials, global weights,
horizon win rates, baselines, a JSON summary, and `origin_quality_audit.csv`.
Before optimization, a whole origin is automatically removed when any horizon
has forecast coverage below 80% of the median daily universe or fewer than 30
positive candidates. Each notebook packages the results and automatically
starts downloading its ZIP when the run finishes.

These runs calibrate a secondary screener around already-trained frozen models.
They are not independent evaluations of the underlying Kronos checkpoints for
dates that were included in those checkpoints' fine-tuning data. The temporal
holdout measures the scorer weights, while genuinely live dates after the model
cutoff remain the strongest final check.

Regenerate all notebooks with:

```bash
python3 "BackTest Kronos Screener/build_kaggle_notebooks.py"
```
