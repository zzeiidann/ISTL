# Daily Screener — TF15 vs TF1D

Notebook ini membandingkan dua model Kronos untuk 3 Agustus 2026:

- TF15: `production_model` dari `kronos_idx_15m_outputs.zip`, hanya forecast
  bar pertama pukul 09:00 WIB.
- TF1D: predictor dan tokenizer IDX dari run
  `tokenizer-idx-e10-predictor-e4`, hanya forecast harian 3 Agustus.

Keduanya memakai data terakhir sampai 31 Juli 2026 dan menghasilkan Top 30
terpisah serta tabel perbandingan gabungan. Notebook mencari repo, data, dan ZIP
secara otomatis pada local workspace, Kaggle, atau Google Drive/Colab.

Untuk Kaggle, tambahkan repository/dataset ISTL dan upload ZIP 15-menit sebagai
Kaggle Dataset. Untuk Colab, simpan repo dan ZIP di Google Drive atau upload ke
`/content`.

Regenerasikan notebook dengan:

```bash
python3 "Daily Screener/build_daily_screener_notebook.py"
```
