# Kronos-small fine-tuning untuk 50 emiten IDX

Paket ini disiapkan untuk Kaggle GPU:

- `data/idx_kronos_50_daily.parquet`: OHLCV harian 50 emiten, 2021-06-16
  sampai 2026-07-30 (sesuai ketersediaan masing-masing emiten).
- `kronos_idx_kaggle_finetune.ipynb`: fine-tune predictor Kronos-small,
  forecast seluruh emiten, ranking 30 saham, dan dashboard lima saham teratas.
- `Kronos/`: clone source resmi pada commit
  `67b630e67f6a18c9e9be918d9b4337c960db1e9a`.

## Kaggle

1. Buat notebook Kaggle dan aktifkan GPU serta Internet.
2. Clone repository ISTL Anda di `/kaggle/working` dengan
   `git clone --recurse-submodules https://github.com/zzeiidann/ISTL.git`.
3. Buka/upload notebook ini, lalu **Run All**.
4. Output utama akan berada di `/kaggle/working/kronos_idx_outputs/` dan
   arsip `kronos_idx_outputs.zip`.

Notebook membekukan tokenizer dan hanya fine-tune predictor Kronos-small.
Training gradient berhenti pada 2025-12-31; data 2026 hanya dipakai sebagai
context inference agar prinsip untouched 2026 tetap terjaga.

Prediksi adalah keluaran model statistik, bukan rekomendasi investasi.
