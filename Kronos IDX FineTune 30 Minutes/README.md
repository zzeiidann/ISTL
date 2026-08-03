# Kronos IDX fine-tuning — 30-minute bars

Pipeline terpisah untuk fine-tune `Kronos-base` pada OHLCV IDX interval 30
menit. Profil training mengikuti varian 15-menit dan harian 80 GB.

## Window

- `BARS_PER_SESSION = 10`.
- `LOOKBACK = 120` bar, kira-kira 12 sesi perdagangan.
- `PRED_LEN = 10` bar, kira-kira satu sesi forecast.
- Total sequence training 131 bar, aman di bawah `MAX_CONTEXT = 512`.

Jumlah bar aktual dapat berbeda pada Jumat, sesi pendek, suspensi, dan ticker
illiquid. Dataset menggunakan bar valid, bukan durasi kalender mentah.

## Files

- `download_idx_30m.py`: download snapshot seluruh universe ke
  `data/idx_kronos_all_30m.parquet`.
- `build_kaggle_notebook_30m_80gb.py`: generator notebook.
- `kronos_idx_kaggle_finetune_30m_80gb.ipynb`: training, validation, production
  refit, forecast, ranking, dan dashboard untuk Kaggle/Colab.

```bash
python3 "Kronos IDX FineTune 30 Minutes/download_idx_30m.py"
python3 "Kronos IDX FineTune 30 Minutes/build_kaggle_notebook_30m_80gb.py"
```

Output runtime memakai `kronos_idx_30m_outputs`, sehingga tidak menimpa hasil
TF15 atau TF1D.

Bobot training/refit berhenti pada 30 Juli 2026. Inference context dan anchor
close memakai data aktual sampai 31 Juli, sehingga forecast pertama dimulai
3 Agustus tanpa memakai bar aktual 3 Agustus yang sudah ada di snapshot.
