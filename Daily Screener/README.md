# Daily Screener — TF15 vs TF1D

Notebook ini membandingkan dua model Kronos untuk 3 Agustus 2026:

- TF15: `production_model` yang tersimpan di repository melalui Git LFS, hanya
  forecast bar pertama pukul 09:00 WIB. ZIP lama tetap didukung sebagai fallback.
- TF1D: predictor dan tokenizer IDX dari run
  `tokenizer-idx-e10-predictor-e4`, hanya forecast harian 3 Agustus.

Keduanya memakai data terakhir sampai 31 Juli 2026 dan menghasilkan Top 30
terpisah serta tabel perbandingan gabungan. Notebook mencari repo, data, dan ZIP
secara otomatis pada local workspace, Kaggle, atau Google Drive/Colab. Jika repo
belum tersedia, notebook otomatis menjalankan `git clone` dan menarik objek Git
LFS yang diperlukan.

Kaggle/Colab tidak perlu upload ZIP lagi. Notebook otomatis clone repository dan
menarik checkpoint TF15 serta TF1D yang diperlukan lewat selective Git LFS.
Source Kronos resmi juga otomatis di-clone dan dipin ke commit yang digunakan
saat training apabila submodule repo belum tersedia.

Regenerasikan notebook dengan:

```bash
python3 "Daily Screener/build_daily_screener_notebook.py"
```
