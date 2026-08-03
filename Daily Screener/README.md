# Daily Screener — TF15 vs TF1D

Notebook ini membandingkan dua model Kronos untuk 3 Agustus 2026:

- TF15: `production_model` yang tersimpan di repository melalui Git LFS, hanya
  forecast bar pertama pukul 09:00 WIB.
- TF1D: predictor dan tokenizer IDX dari run
  `tokenizer-idx-e10-predictor-e4`, hanya forecast harian 3 Agustus.

Keduanya memakai data terakhir sampai 31 Juli 2026 dan menghasilkan Top 30
untuk 3 dan 4 Agustus serta tabel perbandingan gabungan. Forecast 4 Agustus
dibuat secara causal: TF15 mengambil step 21 setelah membentuk 20 bar tanggal 3,
sedangkan TF1D mengambil step 2. Notebook memakai workspace runtime langsung
(`/kaggle/working`, `/content`, atau working directory lokal), kemudian
clone/pull repository dan menarik objek Git LFS yang diperlukan. Tidak ada
Google Drive mount, pencarian recursive, permission prompt, atau ZIP eksternal.

Notebook otomatis menarik checkpoint TF15 serta TF1D lewat selective Git LFS.
Source Kronos resmi juga otomatis di-clone dan dipin ke commit yang digunakan
saat training apabila submodule repo belum tersedia.

Notebook memasang PyTorch 2.7.1 CUDA 12.8 agar kompatibel dengan GPU Blackwell
`sm_120`, lalu menjalankan architecture smoke test. Jika torch lama sudah pernah
di-import dalam session Kaggle, restart session sekali setelah cell instalasi.

Regenerasikan notebook dengan:

```bash
python3 "Daily Screener/build_daily_screener_notebook.py"
```
