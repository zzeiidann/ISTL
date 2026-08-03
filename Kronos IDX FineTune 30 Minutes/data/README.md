# 30-minute data

`download_idx_30m.py` membuat:

- `idx_kronos_all_30m.parquet`
- `missing_30m.csv`

Schema Parquet: `ticker`, `date`, `open`, `high`, `low`, `close`, `volume`, dan
`amount`. Timestamp disimpan timezone-naive dalam waktu Asia/Jakarta.
