# Intraday data

Run `../download_idx_15m.py` to create:

- `idx_kronos_all_15m.parquet`
- `missing_15m.csv`

Expected Parquet columns are `ticker`, `date`, `open`, `high`, `low`, `close`,
`volume`, and `amount`. Timestamps are timezone-naive Asia/Jakarta market time.

The raw intraday snapshot is intentionally not generated during notebook
training. Keeping it as a separate immutable input makes validation and future
comparisons reproducible.
