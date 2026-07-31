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
Training target berakhir sebelum 1 Juli 2026, sedangkan validation target
memakai 1–30 Juli 2026.

Snapshot ini memakai completed close `AS_OF_DATE = 2026-07-30`. Day 1 forecast
dimulai pada 31 Juli dan dibandingkan dengan close terakhir pada atau sebelum
30 Juli.

Prediksi adalah keluaran model statistik, bukan rekomendasi investasi.

## Varian GPU 80 GB

`kronos_idx_kaggle_finetune_80gb.ipynb` adalah notebook terpisah untuk GPU
A100/H100 80 GB. Varian ini memakai 200.000 dynamic, recency-aware windows per
epoch, BF16, batch 128, DataLoader notebook-safe, seluruh validation Juli, dan
clean production refit sampai 30 Juli. Regenerasikan dengan
`build_kaggle_notebook_80gb.py`; notebook
standar tidak diubah oleh konfigurasi training varian ini.

Output varian 80 GB bersifat runtime-adaptive: Kaggle memakai
`/kaggle/working/kronos_idx_outputs`, Colab memakai Google Drive jika sudah
mounted di `/content/drive/MyDrive`, dengan fallback `/content`, dan runtime
lokal memakai direktori project. Checkpoint terbaik dan final production model
disimpan di bawah output directory tersebut.
