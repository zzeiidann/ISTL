# Kronos-base fine-tuning untuk seluruh universe IDX

Paket ini disiapkan untuk Kaggle GPU:

- `data/idx_kronos_all_daily.parquet`: OHLCV harian seluruh ticker yang
  tersedia di Yahoo Finance (958 dari 959 kode), 2021-06-16 sampai 2026-07-30.
- `data/universe_all.csv`: seluruh 959 kode dari `KODE.xlsx`, termasuk flag
  ketersediaan Yahoo Finance.
- `kronos_idx_kaggle_finetune.ipynb`: fine-tune predictor Kronos-base,
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

Notebook mem-pin PyTorch 2.3.1 CUDA 11.8 dan memaksa math SDPA agar kompatibel
dengan Kaggle Tesla P100 maupun T4. Setelah mengganti versi PyTorch atau setelah
CUDA error, restart Kaggle session sebelum menjalankan ulang dari awal.

Notebook membekukan tokenizer dan hanya fine-tune predictor Kronos-base.
Training gradient berhenti pada 2025-12-31; data 2026 hanya dipakai sebagai
context inference agar prinsip untouched 2026 tetap terjaga.

Snapshot ini memakai `AS_OF_DATE = 2026-07-29`. Bar 30 Juli tidak menjadi
anchor karena dapat merupakan bar intraday ketika dataset diambil. Day 1
forecast dimulai pada 30 Juli dan dibandingkan dengan close terakhir pada atau
sebelum 29 Juli.

Prediksi adalah keluaran model statistik, bukan rekomendasi investasi.
