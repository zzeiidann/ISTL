# Kronos IDX fine-tuning — 15-minute bars

Pipeline terpisah untuk mengadaptasi `Kronos-base` pada OHLCV IDX interval
15 menit. Profil optimizer dan training mengikuti notebook 80 GB harian.

## Window

- `BARS_PER_SESSION = 20` sebagai ekuivalensi konservatif satu sesi IDX.
- `LOOKBACK = 240` bar, kira-kira 12 sesi perdagangan.
- `PRED_LEN = 20` bar, kira-kira satu sesi forecast.
- Total sequence training 261 bar (`LOOKBACK + PRED_LEN + 1`), di bawah
  `MAX_CONTEXT = 512`.

Jumlah bar aktual dapat berbeda pada Jumat, sesi pendek, suspensi, atau data
provider yang tidak lengkap. Window model dihitung dalam bar valid, bukan jam
kalender.

## Files

- `download_idx_15m.py`: membangun `data/idx_kronos_all_15m.parquet` dari
  universe harian melalui Yahoo Finance.
- `build_kaggle_notebook_15m_80gb.py`: meregenerasikan notebook.
- `kronos_idx_kaggle_finetune_15m_80gb.ipynb`: notebook training dan forecast.

Jalankan downloader sedekat mungkin dengan tanggal training karena Yahoo
Finance membatasi histori intraday. Untuk run reproducible, simpan snapshot
Parquet dan gunakan file yang sama untuk seluruh eksperimen.

```bash
python3 "Kronos IDX FineTune 15 Minutes/download_idx_15m.py"
python3 "Kronos IDX FineTune 15 Minutes/build_kaggle_notebook_15m_80gb.py"
```

Output notebook disimpan sebagai `kronos_idx_15m_outputs` agar tidak menimpa
hasil model harian.

Checkpoint production hasil run 30 Juli disimpan melalui Git LFS di
`results/2026-07-30/refit-run-e4/production_model` dan dapat langsung digunakan
oleh `Daily Screener` tanpa ZIP eksternal.
